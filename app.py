"""
Interfaz web para el sistema RAG multimodal.
Ejecutar con: python app.py
Abrir en: http://localhost:5001
"""

import sys
import os
import io
import json
import time
from pathlib import Path
from contextlib import redirect_stdout
from unittest import result


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from flask import Flask, render_template_string, request, jsonify
import yaml

app = Flask(__name__)

with open("config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

VECTOR_STORE_PATH = CONFIG["paths"]["vector_store"]
EXPERIMENTS_DIR = CONFIG["paths"]["experiments"]

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        from src.embeddings.vector_store import load_vector_store
        _vector_store = load_vector_store(VECTOR_STORE_PATH)
    return _vector_store


def deduplicate_sources(sources: list) -> list:
    seen = set()
    unique = []

    for s in sources or []:
        key = (
            s.get("source_file", "desconocido"),
            s.get("page", "N/A"),
            s.get("type", "texto")
        )
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique

def save_live_metrics(query, mode, result):
    from src.evaluation.metrics import compute_all_metrics

    metrics_path = Path("data/live_metrics.json")
    metrics_path.parent.mkdir(exist_ok=True)

    previous = []
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            previous = json.load(f)

    entry = {
        "timestamp": time.strftime("%d/%m/%Y %H:%M:%S"),
        "question": query,
        "mode": mode,
        "rag_answer": result.get("rag_answer", ""),
        "baseline_answer": result.get("baseline_answer", ""),
        "rag_sources": result.get("rag_sources", []),
    }

    if result.get("rag_answer"):
        metrics = compute_all_metrics(
            result["rag_answer"],
            rag_sources=result.get("rag_sources", []),
            reference_answer=None
        )

        context_text = " ".join(
            s.get("content_preview", "")
            for s in result.get("rag_sources", [])
        )

        answer_words = set(result["rag_answer"].lower().split())
        context_words = set(context_text.lower().split())

        if answer_words and context_words:
            overlap = len(answer_words & context_words) / len(answer_words)
        else:
            overlap = 0

        metrics["faithfulness"] = overlap
        metrics["rouge1_vs_context"] = overlap

        entry["rag_metrics"] = metrics

    if result.get("baseline_answer"):
        entry["baseline_metrics"] = compute_all_metrics(
            result["baseline_answer"],
            reference_answer=None
        )

    previous.append(entry)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(previous, f, indent=2, ensure_ascii=False)

HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lab IA Generativa — RAG Multimodal</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#0f0f0f; --surface:#1a1a1a; --surface2:#242424; --border:#2e2e2e;
  --accent:#e8d5a3; --accent2:#7eb8a4; --accent3:#c4856a;
  --text:#e8e8e8; --text-dim:#888;
  --rag:#7eb8a4; --bl:#c4856a;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh}
header{border-bottom:1px solid var(--border);padding:1.2rem 2.5rem;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;background:rgba(15,15,15,.95);backdrop-filter:blur(12px);z-index:100}
.logo{font-family:'DM Serif Display',serif;font-size:1.2rem;color:var(--accent)}
.logo span{font-style:italic;color:var(--text-dim);font-size:.85rem;margin-left:.5rem}
.dot{width:8px;height:8px;border-radius:50%;background:var(--accent2);box-shadow:0 0 8px var(--accent2);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.tabs{display:flex;border-bottom:1px solid var(--border);padding:0 2.5rem;background:var(--surface)}
.tab{padding:.85rem 1.4rem;cursor:pointer;font-size:.8rem;font-weight:500;color:var(--text-dim);border-bottom:2px solid transparent;transition:all .2s;letter-spacing:.05em;text-transform:uppercase;font-family:'DM Mono',monospace}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover:not(.active){color:var(--text)}
.panel{display:none;padding:2.5rem;max-width:1100px;margin:0 auto}
.panel.active{display:block}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:1.25rem}
.card-head{padding:.9rem 1.25rem;display:flex;align-items:center;gap:.6rem;border-bottom:1px solid var(--border)}
.card-body{padding:1.25rem}
.badge{font-family:'DM Mono',monospace;font-size:.7rem;font-weight:500;letter-spacing:.08em;text-transform:uppercase;padding:.25rem .6rem;border-radius:4px}
.badge-rag{background:rgba(126,184,164,.15);color:var(--rag)}
.badge-bl{background:rgba(196,133,106,.15);color:var(--bl)}
.badge-accent{background:rgba(232,213,163,.15);color:var(--accent)}
textarea{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:1rem;padding:1rem;resize:none;outline:none;line-height:1.6;transition:border-color .2s;min-height:80px}
textarea:focus{border-color:var(--accent)}
.btn{padding:.65rem 1.4rem;border-radius:8px;border:none;cursor:pointer;font-family:'DM Mono',monospace;font-size:.78rem;font-weight:500;letter-spacing:.05em;transition:all .2s;text-transform:uppercase}
.btn-primary{background:var(--accent);color:#0f0f0f}
.btn-primary:hover{background:#f0ddb5;transform:translateY(-1px)}
.btn-secondary{background:transparent;color:var(--text-dim);border:1px solid var(--border)}
.btn-secondary:hover{color:var(--text);border-color:var(--text-dim)}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-row{display:flex;gap:.75rem;margin-top:1rem;flex-wrap:wrap}
.mono{font-family:'DM Mono',monospace}
.dim{color:var(--text-dim)}
.small{font-size:.78rem}
.label{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}
@media(max-width:700px){.grid2{grid-template-columns:1fr}}
.loading{display:none;align-items:center;gap:.75rem;padding:.75rem 0;color:var(--text-dim);font-family:'DM Mono',monospace;font-size:.78rem}
.loading.show{display:flex}
.spinner{width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.answer{font-size:.95rem;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.sources{margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border)}
.chip{display:inline-flex;align-items:center;gap:.3rem;background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:.2rem .5rem;font-family:'DM Mono',monospace;font-size:.68rem;color:var(--text-dim);margin:.15rem}
.chip.visual{border-color:rgba(126,184,164,.3);color:var(--rag)}
.suggestions{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.75rem}
.sug{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:.35rem .75rem;font-size:.8rem;color:var(--text-dim);cursor:pointer;transition:all .2s}
.sug:hover{color:var(--accent);border-color:var(--accent)}
.mode-toggle{display:flex;gap:.5rem;align-items:center;margin-top:.75rem}
.tog{padding:.3rem .7rem;border-radius:5px;border:1px solid var(--border);background:transparent;color:var(--text-dim);font-family:'DM Mono',monospace;font-size:.7rem;cursor:pointer;transition:all .2s}
.tog.active{background:var(--surface2);color:var(--accent);border-color:var(--accent)}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.25rem}
.metric-val{font-family:'DM Serif Display',serif;font-size:2rem;line-height:1}
.metric-cmp{font-size:.73rem;color:var(--text-dim);margin-top:.4rem;font-family:'DM Mono',monospace}
.better{color:var(--accent2)}
.worse{color:var(--accent3)}
.bar-row{display:flex;align-items:center;gap:1rem;margin-bottom:.6rem}
.bar-lbl{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--text-dim);width:80px;flex-shrink:0;text-align:right}
.bar-track{flex:1;height:7px;background:var(--surface2);border-radius:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.bar-fill.rag{background:var(--rag)}
.bar-fill.bl{background:var(--bl);opacity:.7}
.bar-val{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--text-dim);width:48px}
.section-title{font-family:'DM Serif Display',serif;font-size:1.35rem;color:var(--accent);margin-bottom:1.25rem;padding-bottom:.75rem;border-bottom:1px solid var(--border)}
.empty{text-align:center;padding:4rem 2rem;color:var(--text-dim);font-family:'DM Serif Display',serif;font-size:1.1rem;font-style:italic}
.log{display:none;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.25rem;font-family:'DM Mono',monospace;font-size:.76rem;color:var(--text-dim);line-height:1.8;max-height:320px;overflow-y:auto;margin-top:1rem}
.filter-row{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.5rem}
.anim{animation:slideUp .3s ease}
@keyframes slideUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.badge-agent{background:rgba(180,160,220,.15);color:#b4a0dc}
.step{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:.65rem .9rem;margin:.35rem 0;font-size:.76rem;font-family:'DM Mono',monospace}
.step-tool{color:#b4a0dc;font-weight:500}
.step-input{color:var(--text-dim);margin-top:.2rem}
.step-result{color:var(--text-dim);margin-top:.2rem;opacity:.7}
</style>
</head>
<body>

<header>
  <div class="logo">RAG Multimodal <span>Lab IA Generativa</span></div>
  <div class="dot"></div>
</header>

<div class="tabs">
  <div class="tab active" onclick="goTab('consulta',this)">Consulta</div>
  <div class="tab" onclick="goTab('comparacion',this)">Comparación</div>
  <div class="tab" onclick="goTab('metricas',this)">Métricas</div>
  <div class="tab" onclick="goTab('indice',this)">Índice</div>
</div>

<div id="panel-consulta" class="panel active">
  <div class="card">
    <div class="card-body">
      <div class="label">Pregunta</div>
      <textarea id="query" placeholder="Escribe tu pregunta sobre los documentos..." rows="3"></textarea>
      <div class="mode-toggle">
        <span class="label" style="margin:0">Modo:</span>
        <button class="tog active" onclick="setMode('rag',this)">RAG</button>
        <button class="tog" onclick="setMode('baseline',this)">Baseline</button>
        <button class="tog" onclick="setMode('ambos',this)">Ambos</button>
        <button class="tog" onclick="setMode('agente',this)">Agente</button>
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" id="btn-ask" onclick="doAsk()">Preguntar</button>
        <button class="btn btn-secondary" onclick="document.getElementById('query').value='';document.getElementById('res-consulta').innerHTML=''">Limpiar</button>
      </div>
      <div class="suggestions">
        <div class="sug" onclick="document.getElementById('query').value=this.textContent">¿Qué es RAG?</div>
        <div class="sug" onclick="document.getElementById('query').value=this.textContent">¿Qué tipos de RAG existen?</div>
        <div class="sug" onclick="document.getElementById('query').value=this.textContent">¿Qué es BLIP-2?</div>
        <div class="sug" onclick="document.getElementById('query').value=this.textContent">¿Qué es el Q-Former?</div>
        <div class="sug" onclick="document.getElementById('query').value=this.textContent">¿Qué mide RAGAS?</div>
        <div class="sug" onclick="document.getElementById('query').value=this.textContent">¿Qué ejemplos visuales hay en los papers?</div>
      </div>
    </div>
  </div>
  <div class="loading" id="load-ask"><div class="spinner"></div><span>Consultando...</span></div>
  <div id="res-consulta"></div>
</div>

<div id="panel-comparacion" class="panel">
  <div class="section-title">Resultados del dataset de evaluación</div>
  <div class="filter-row">
    <button class="btn btn-secondary" id="f-todas" style="border-color:var(--accent);color:var(--accent)" onclick="setFilter('todas',this)">Todas</button>
    <button class="btn btn-secondary" id="f-teoria" onclick="setFilter('teoria',this)">Teoría</button>
    <button class="btn btn-secondary" id="f-multimodalidad" onclick="setFilter('multimodalidad',this)">Multimodal</button>
    <button class="btn btn-secondary" id="f-evaluacion" onclick="setFilter('evaluacion',this)">Evaluación</button>
    <button class="btn btn-secondary" id="f-comparacion" onclick="setFilter('comparacion',this)">Comparación</button>
  </div>
  <div id="res-comp"><div class="empty">Cargando resultados...</div></div>
</div>

<div id="panel-metricas" class="panel">
  <div id="res-metrics"><div class="empty">Cargando métricas...</div></div>
</div>

<div id="panel-indice" class="panel">
  <div class="section-title">Gestión del índice</div>
  <div id="indice-status" style="margin-bottom:2rem"></div>

  <!-- Documentos indexados -->
  <div class="card" style="margin-bottom:2rem">
    <div class="card-head">
      <span class="badge badge-accent">Documentos indexados</span>
      <span class="dim small mono" id="idx-summary" style="margin-left:auto"></span>
    </div>
    <div class="card-body" id="indexed-docs-list">
      <div class="dim small">Cargando...</div>
    </div>
  </div>

  <!-- Añadir documento -->
  <div class="card" style="margin-bottom:2rem">
    <div class="card-head"><span class="badge badge-rag">Añadir documento</span></div>
    <div class="card-body">
      <div class="answer" style="font-size:.88rem;margin-bottom:1rem">
        Sube un PDF nuevo para añadirlo al índice de forma incremental, sin regenerar los embeddings existentes.
      </div>
      <div style="display:flex;gap:.75rem;align-items:center;flex-wrap:wrap">
        <label class="btn btn-secondary" style="cursor:pointer">
          Seleccionar PDF
          <input type="file" id="pdf-upload" accept=".pdf" style="display:none" onchange="onFileSelected(this)">
        </label>
        <span class="dim small mono" id="selected-file">Ningún archivo seleccionado</span>
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" id="btn-add-doc" onclick="addDocument()" disabled>Añadir al índice</button>
      </div>
      <div class="loading" id="load-add"><div class="spinner"></div><span>Procesando documento...</span></div>
      <div class="log" id="log-add"></div>
    </div>
  </div>

  <!-- Reconstruir índice completo -->
  <div class="grid2" style="margin-bottom:2rem">
    <div class="card">
      <div class="card-head"><span class="badge badge-rag">Reconstruir · Solo texto</span></div>
      <div class="card-body">
        <div class="answer" style="font-size:.88rem;margin-bottom:1rem">Regenera todo el índice desde cero con todos los PDFs de data/raw. Borra los embeddings existentes.</div>
        <button class="btn btn-primary" id="btn-idx-texto" onclick="buildIdx('texto')">Reconstruir índice de texto</button>
      </div>
    </div>
    <div class="card">
      <div class="card-head"><span class="badge badge-accent">Reconstruir · Texto + Imágenes</span></div>
      <div class="card-body">
        <div class="answer" style="font-size:.88rem;margin-bottom:1rem">Regenera índice con texto + captions de imágenes (LLaVA). Requiere Ollama.</div>
        <button class="btn btn-secondary" id="btn-idx-multi" onclick="buildIdx('multimodal')">Reconstruir índice multimodal</button>
      </div>
    </div>
  </div>

  <div class="loading" id="load-idx"><div class="spinner"></div><span id="load-idx-txt">Construyendo índice...</span></div>
  <div class="log" id="log-idx"></div>

  <div style="border-top:1px solid var(--border);padding-top:2rem;margin-top:2rem">
    <div class="section-title" style="font-size:1.1rem">Comparación RAG vs Baseline</div>
    <div class="answer" style="font-size:.88rem;color:var(--text-dim);margin-bottom:1.25rem">
      Ejecuta el dataset completo de evaluación, calcula métricas ROUGE y guarda los resultados en
      <span class="mono" style="color:var(--accent)">experiments/</span>.
    </div>
    <div class="grid2" style="margin-bottom:1.25rem">
      <div class="metric-card">
        <div class="label">RAG · ROUGE-1</div>
        <div class="metric-val" style="color:var(--rag)" id="cmp-rag-r1">—</div>
      </div>
      <div class="metric-card">
        <div class="label">Baseline · ROUGE-1</div>
        <div class="metric-val" style="color:var(--bl)" id="cmp-bl-r1">—</div>
      </div>
    </div>
    <button class="btn btn-primary" id="btn-run-cmp" onclick="runCmp()">Ejecutar comparación</button>
    <div class="loading" id="load-cmp"><div class="spinner"></div><span>Procesando preguntas... puede tardar varios minutos.</span></div>
    <div class="log" id="log-cmp"></div>
  </div>
</div>

<script>
var curMode = 'rag';
var compData = [];
var compFilter = 'todas';

function goTab(name, el) {
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active')});
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active')});
  el.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
  if (name === 'metricas') loadMetrics();
  if (name === 'comparacion') loadComp();
  if (name === 'indice') loadIdxStatus();
}

function setMode(m, btn) {
  curMode = m;
  document.querySelectorAll('.tog').forEach(function(b){b.classList.remove('active')});
  btn.classList.add('active');
}

function setFilter(cat, btn) {
  compFilter = cat;
  document.querySelectorAll('.filter-row .btn').forEach(function(b){
    b.style.borderColor = '';
    b.style.color = '';
  });
  btn.style.borderColor = 'var(--accent)';
  btn.style.color = 'var(--accent)';
  renderComp();
}

function doAsk() {
  var q = document.getElementById('query').value.trim();
  if (!q) return;
  var btn = document.getElementById('btn-ask');
  btn.disabled = true;
  document.getElementById('load-ask').classList.add('show');
  document.getElementById('res-consulta').innerHTML = '';

  fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: q, mode: curMode})
  }).then(function(r){return r.json()}).then(function(data){
    renderConsulta(data);
  }).catch(function(){
    document.getElementById('res-consulta').innerHTML = '<div class="empty">Error al conectar con el servidor.</div>';
  }).finally(function(){
    btn.disabled = false;
    document.getElementById('load-ask').classList.remove('show');
  });
}

function renderConsulta(data) {
  var cont = document.getElementById('res-consulta');

  if (data.error) {
    cont.innerHTML =
      '<div class="card anim">' +
        '<div class="card-head"><span class="badge badge-bl">ERROR</span></div>' +
        '<div class="card-body"><div class="answer" style="color:#c4856a;">' + data.error + '</div></div>' +
      '</div>';
    return;
  }

  if (curMode === 'ambos') {
    cont.innerHTML = buildGrid(data.rag_answer, data.rag_sources, data.baseline_answer, null);
    return;
  }

  if (curMode === 'agente') {
    cont.innerHTML =
      '<div class="card anim">' +
        '<div class="card-head"><span class="badge badge-agent">AGENTE</span></div>' +
        '<div class="card-body">' +
          '<div class="answer">' + (data.rag_answer || 'Sin respuesta.') + '</div>' +
          stepsHtml(data.agent_steps || []) +
          srcHtml(data.rag_sources || []) +
        '</div>' +
      '</div>';
    return;
  }

  var isRag = curMode === 'rag';
  var ans = isRag ? data.rag_answer : data.baseline_answer;
  var src = isRag ? (data.rag_sources || []) : [];
  var badgeCls = isRag ? 'badge-rag' : 'badge-bl';
  var lbl = isRag ? 'RAG' : 'BASELINE';

  cont.innerHTML =
    '<div class="card anim">' +
      '<div class="card-head"><span class="badge ' + badgeCls + '">' + lbl + '</span></div>' +
      '<div class="card-body">' +
        '<div class="answer">' + (ans || 'Sin respuesta.') + '</div>' +
        srcHtml(src) +
      '</div>' +
    '</div>';
}

function buildGrid(ragAns, ragSrc, blAns, blSrc) {
  return '<div class="grid2">' +
    '<div class="card anim"><div class="card-head"><span class="badge badge-rag">RAG</span></div><div class="card-body"><div class="answer">' + (ragAns || 'Sin respuesta.') + '</div>' + srcHtml(ragSrc || []) + '</div></div>' +
    '<div class="card anim"><div class="card-head"><span class="badge badge-bl">Baseline</span></div><div class="card-body"><div class="answer">' + (blAns || 'Sin respuesta.') + '</div></div></div>' +
  '</div>';
}

function stepsHtml(steps) {
  if (!steps || !steps.length) return '';
  var rows = steps.map(function(s, i) {
    return '<div class="step">' +
      '<div class="step-tool">&#x2192; ' + (i+1) + '. ' + s.tool + '</div>' +
      '<div class="step-input">Input: ' + s.input + '</div>' +
      (s.result_preview ? '<div class="step-result">' + s.result_preview.substring(0,200) + '…</div>' : '') +
    '</div>';
  }).join('');
  return '<div class="sources"><div class="label">Pasos del agente</div>' + rows + '</div>';
}

function srcHtml(sources) {
  if (!sources || !sources.length) return '';

  const seen = new Set();
  const unique = [];

  sources.forEach(function(s) {
    const key = `${s.source_file}||${s.page}||${s.type || 'text'}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(s);
    }
  });

  var chips = unique.map(function(s){
    var isVis = s.type === 'image_caption';
    return '<span class="chip' + (isVis ? ' visual' : '') + '">' + (isVis ? '🖼' : '📄') + ' ' + s.source_file + ' · pág. ' + s.page + '</span>';
  }).join('');

  return '<div class="sources"><div class="label">Fuentes</div>' + chips + '</div>';
}

function loadComp() {
  if (compData.length > 0) { renderComp(); return; }
  fetch('/comparison').then(function(r){return r.json()}).then(function(d){
    compData = d;
    renderComp();
  }).catch(function(){
    document.getElementById('res-comp').innerHTML = '<div class="empty">No hay resultados. Ejecuta primero la comparación desde la pestaña Índice.</div>';
  });
}

function renderComp() {
  var cont = document.getElementById('res-comp');
  var data = compFilter === 'todas' ? compData : compData.filter(function(d){return d.category === compFilter});
  if (!data.length) {
    cont.innerHTML = '<div class="empty">No hay preguntas en esta categoría.</div>';
    return;
  }

  cont.innerHTML = data.map(function(item){
    var rm = item.rag_metrics || {};
    var bm = item.baseline_metrics || {};
    var r1 = rm.rouge1_f1 || 0;
    var b1 = bm.rouge1_f1 || 0;
    var ragWins = r1 >= b1;
    var rColor = ragWins ? 'var(--accent2)' : 'var(--accent3)';
    var bColor = !ragWins ? 'var(--accent2)' : 'var(--accent3)';

    var seen = new Set();
    var uniqueSources = [];
    (item.rag_sources || []).forEach(function(s){
      var key = `${s.source_file}||${s.page}||${s.type || 'text'}`;
      if (!seen.has(key)) {
        seen.add(key);
        uniqueSources.push(s);
      }
    });

    var chips = uniqueSources.map(function(s){
      var isVis = s.type === 'image_caption';
      return '<span class="chip' + (isVis ? ' visual' : '') + '">' + (isVis ? '🖼' : '📄') + ' ' + s.source_file + ' · pág. ' + s.page + '</span>';
    }).join('');

    return '<div class="card" style="margin-bottom:1.25rem">' +
      '<div class="card-head" style="justify-content:space-between">' +
        '<div style="display:flex;align-items:center;gap:.75rem">' +
          '<span class="mono dim small">#' + item.id + '</span>' +
          '<span style="font-size:.92rem;font-weight:500">' + item.question + '</span>' +
        '</div>' +
        '<span class="badge" style="background:var(--surface2);color:var(--text-dim);flex-shrink:0">' + item.category + '</span>' +
      '</div>' +
      '<div class="grid2" style="padding:1.25rem;gap:1rem">' +
        '<div>' +
          '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem">' +
            '<span class="badge badge-rag">RAG</span>' +
            '<span class="mono small" style="color:' + rColor + '">ROUGE-1: ' + r1.toFixed(3) + '</span>' +
          '</div>' +
          '<div class="answer" style="font-size:.88rem">' + item.rag_answer + '</div>' +
          (chips ? '<div style="margin-top:.75rem">' + chips + '</div>' : '') +
        '</div>' +
        '<div>' +
          '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem">' +
            '<span class="badge badge-bl">Baseline</span>' +
            '<span class="mono small" style="color:' + bColor + '">ROUGE-1: ' + b1.toFixed(3) + '</span>' +
          '</div>' +
          '<div class="answer" style="font-size:.88rem">' + item.baseline_answer + '</div>' +
        '</div>' +
      '</div>' +
    '</div>';
  }).join('');
}

function loadMetrics() {
  fetch('/metrics').then(function(r){return r.json()}).then(function(data){
    if (data.error) {
      document.getElementById('res-metrics').innerHTML =
        '<div class="empty">No hay métricas. Haz una pregunta o ejecuta RAGAS primero.</div>';
      return;
    }

    renderMetrics(data.items || [], data.ragas || {});
  }).catch(function(){
    document.getElementById('res-metrics').innerHTML =
      '<div class="empty">No hay métricas disponibles.</div>';
  });
}

function renderMetrics(items, ragas) {
  var cont = document.getElementById('res-metrics');
  var html = '';

  if (items.length) {
    var last = items[items.length - 1];
    var rag = last.rag_metrics || {};
    var bl = last.baseline_metrics || {};

    html +=
      '<div class="section-title">Métricas de la última consulta</div>' +
      '<div class="card" style="margin-bottom:1.5rem">' +
        '<div class="card-head"><span class="badge badge-accent">Pregunta</span></div>' +
        '<div class="card-body">' +
          '<div class="answer">' + last.question + '</div>' +
          '<div class="mono small dim" style="margin-top:.75rem">' + last.timestamp + '</div>' +
        '</div>' +
      '</div>' +
      '<div class="grid2" style="margin-bottom:2rem">';

    if (last.rag_answer) {
      html +=
        '<div class="metric-card"><div class="label">RAG · Faithfulness</div><div class="metric-val" style="color:var(--rag)">' + ((rag.faithfulness || 0).toFixed(3)) + '</div></div>' +
        '<div class="metric-card"><div class="label">RAG · ROUGE contexto</div><div class="metric-val" style="color:var(--rag)">' + ((rag.rouge1_vs_context || 0).toFixed(3)) + '</div></div>' +
        '<div class="metric-card"><div class="label">RAG · Longitud</div><div class="metric-val" style="color:var(--rag)">' + (rag.answer_length_words || 0) + '</div></div>' +
        '<div class="metric-card"><div class="label">RAG · Sin respuesta</div><div class="metric-val" style="color:var(--rag)">' + ((rag.mentions_no_info || 0) * 100).toFixed(0) + '%</div></div>';
    }

    if (last.baseline_answer) {
      html +=
        '<div class="metric-card"><div class="label">Baseline · Longitud</div><div class="metric-val" style="color:var(--bl)">' + (bl.answer_length_words || 0) + '</div></div>' +
        '<div class="metric-card"><div class="label">Baseline · Sin respuesta</div><div class="metric-val" style="color:var(--bl)">' + ((bl.mentions_no_info || 0) * 100).toFixed(0) + '%</div></div>';
    }

    html += '</div>';
  }

  if (ragas && ragas.n_questions) {
    html +=
      '<div class="section-title">Métricas RAGAS</div>' +
      '<div class="grid2" style="margin-bottom:2rem">' +
        '<div class="metric-card"><div class="label">RAGAS · Faithfulness</div><div class="metric-val" style="color:var(--accent2)">' + (ragas.faithfulness || 0).toFixed(3) + '</div></div>' +
        '<div class="metric-card"><div class="label">RAGAS · Answer Relevance</div><div class="metric-val" style="color:var(--accent2)">' + (ragas.answer_relevancy || 0).toFixed(3) + '</div></div>' +
        '<div class="metric-card"><div class="label">RAGAS · Context Precision</div><div class="metric-val" style="color:var(--accent)">' + (ragas.context_precision || 0).toFixed(3) + '</div></div>' +
        '<div class="metric-card"><div class="label">RAGAS · Context Recall</div><div class="metric-val" style="color:var(--accent)">' + (ragas.context_recall || 0).toFixed(3) + '</div></div>' +
      '</div>' +
      '<div class="mono small dim" style="margin-bottom:2rem">RAGAS calculado sobre ' + ragas.n_questions + ' preguntas del dataset de evaluación.</div>';
  }

  if (items.length) {
    html += '<div class="section-title">Histórico de consultas</div>';

    items.slice().reverse().forEach(function(item){
      var rm = item.rag_metrics || {};
      var bm = item.baseline_metrics || {};

      html +=
        '<div class="card">' +
          '<div class="card-head" style="justify-content:space-between">' +
            '<span class="answer" style="font-size:.9rem">' + item.question + '</span>' +
            '<span class="mono small dim">' + item.timestamp + '</span>' +
          '</div>' +
          '<div class="card-body">' +
            '<div class="grid2">' +
              '<div>' +
                '<span class="badge badge-rag">RAG</span>' +
                '<div class="mono small dim" style="margin-top:.6rem">Faithfulness: ' + ((rm.faithfulness || 0).toFixed(3)) + '</div>' +
                '<div class="mono small dim">ROUGE contexto: ' + ((rm.rouge1_vs_context || 0).toFixed(3)) + '</div>' +
                '<div class="mono small dim">Longitud: ' + (rm.answer_length_words || 0) + '</div>' +
                '<div class="mono small dim">Sin respuesta: ' + ((rm.mentions_no_info || 0) * 100).toFixed(0) + '%</div>' +
              '</div>' +
              '<div>' +
                '<span class="badge badge-bl">Baseline</span>' +
                '<div class="mono small dim" style="margin-top:.6rem">Longitud: ' + (bm.answer_length_words || 0) + '</div>' +
                '<div class="mono small dim">Sin respuesta: ' + ((bm.mentions_no_info || 0) * 100).toFixed(0) + '%</div>' +
              '</div>' +
            '</div>' +
          '</div>' +
        '</div>';
    });
  }

  if (!html) {
    html = '<div class="empty">No hay métricas. Haz una pregunta o ejecuta RAGAS primero.</div>';
  }

  cont.innerHTML = html;
}



function loadIdxStatus() {
  fetch('/index_status').then(function(r){return r.json()}).then(function(data){
    var el = document.getElementById('indice-status');
    var color = data.exists ? 'var(--accent2)' : 'var(--accent3)';
    var icon = data.exists ? '✓' : '✗';
    var extra = data.exists ?
      '<div class="metric-card" style="flex:1;min-width:160px"><div class="label">Última modificación</div><div class="mono" style="font-size:.85rem;margin-top:.5rem;color:var(--text)">' + data.modified + '</div></div>' +
      '<div class="metric-card" style="flex:1;min-width:160px"><div class="label">Tamaño</div><div class="mono" style="font-size:.85rem;margin-top:.5rem;color:var(--text)">' + data.type + '</div></div>' : '';
    el.innerHTML = '<div style="display:flex;gap:1rem;flex-wrap:wrap">' +
      '<div class="metric-card" style="flex:1;min-width:160px"><div class="label">Estado</div><div class="metric-val" style="color:' + color + ';font-size:1.4rem">' + icon + ' ' + (data.exists ? 'Disponible' : 'No existe') + '</div></div>' +
      extra + '</div>';
  });
  loadIndexedSources();
}

function loadIndexedSources() {
  fetch('/indexed_sources').then(function(r){return r.json()}).then(function(data){
    var cont = document.getElementById('indexed-docs-list');
    var summary = document.getElementById('idx-summary');

    if (data.error || !data.sources || data.sources.length === 0) {
      cont.innerHTML = '<div class="dim small" style="font-style:italic">No hay documentos indexados todavía.</div>';
      summary.textContent = '';
      return;
    }

    summary.textContent = data.total_documents + ' docs · ' + data.total_chunks + ' chunks';

    var html = '';
    data.sources.forEach(function(src) {
      var types = src.types.join(', ');
      var pageCount = src.pages.length;

      html += '<div style="display:flex;align-items:center;justify-content:space-between;padding:.6rem 0;border-bottom:1px solid var(--border)">' +
        '<div>' +
          '<div class="mono small" style="color:var(--text)">' + src.source_file + '</div>' +
          '<div class="dim" style="font-size:.72rem">' + src.chunk_count + ' chunks · ' + pageCount + ' páginas · ' + types + '</div>' +
        '</div>' +
        '<button class="btn btn-secondary" style="padding:.25rem .6rem;font-size:.68rem" onclick="deleteSource(\'' + src.source_file.replace(/'/g, "\\'") + '\')">Eliminar</button>' +
      '</div>';
    });

    cont.innerHTML = html;
  }).catch(function(){
    document.getElementById('indexed-docs-list').innerHTML = '<div class="dim small">Error al cargar documentos indexados.</div>';
  });
}

function onFileSelected(input) {
  var lbl = document.getElementById('selected-file');
  var btn = document.getElementById('btn-add-doc');

  if (input.files && input.files.length > 0) {
    lbl.textContent = input.files[0].name;
    lbl.style.color = 'var(--accent)';
    btn.disabled = false;
  } else {
    lbl.textContent = 'Ningún archivo seleccionado';
    lbl.style.color = '';
    btn.disabled = true;
  }
}

function addDocument() {
  var fileInput = document.getElementById('pdf-upload');
  if (!fileInput.files || fileInput.files.length === 0) return;

  var formData = new FormData();
  formData.append('file', fileInput.files[0]);

  var btn = document.getElementById('btn-add-doc');
  btn.disabled = true;
  document.getElementById('load-add').classList.add('show');
  document.getElementById('log-add').style.display = 'none';

  fetch('/add_document', {
    method: 'POST',
    body: formData
  }).then(function(r){return r.json()}).then(function(data){
    var log = document.getElementById('log-add');
    log.style.display = 'block';

    if (data.error) {
      log.innerHTML = '<span style="color:var(--accent3)">ERROR: ' + data.error + '</span>';
    } else {
      log.innerHTML = data.log.replace(/\n/g, '<br>');
    }
    log.scrollTop = log.scrollHeight;

    // Refresh lists
    loadIdxStatus();

    // Reset file input
    fileInput.value = '';
    document.getElementById('selected-file').textContent = 'Ningún archivo seleccionado';
    document.getElementById('selected-file').style.color = '';
  }).catch(function(){
    var log = document.getElementById('log-add');
    log.style.display = 'block';
    log.textContent = 'Error de conexión al añadir el documento.';
  }).finally(function(){
    document.getElementById('load-add').classList.remove('show');
    btn.disabled = true;
  });
}

function deleteSource(sourceFile) {
  if (!confirm('¿Eliminar "' + sourceFile + '" del índice?')) return;

  fetch('/delete_source', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source_file: sourceFile})
  }).then(function(r){return r.json()}).then(function(data){
    if (data.error) {
      alert('Error: ' + data.error);
    }
    loadIndexedSources();
    loadIdxStatus();
  }).catch(function(){
    alert('Error de conexión al eliminar el documento.');
  });
}

function buildIdx(mode) {
  var ids = ['btn-idx-texto','btn-idx-multi','btn-run-cmp'];
  ids.forEach(function(id){document.getElementById(id).disabled = true});
  document.getElementById('load-idx').classList.add('show');
  document.getElementById('load-idx-txt').textContent = mode === 'multimodal'
    ? 'Construyendo índice multimodal con LLaVA... puede tardar 10-20 min.'
    : 'Construyendo índice de texto... puede tardar 2-3 min.';
  document.getElementById('log-idx').style.display = 'none';

  fetch('/build_index', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:mode})
  })
    .then(function(r){return r.json()}).then(function(data){
      var log = document.getElementById('log-idx');
      log.style.display = 'block';
      log.innerHTML = data.log.replace(/\n/g,'<br>');
      log.scrollTop = log.scrollHeight;
      loadIdxStatus();
    }).catch(function(){
      var log = document.getElementById('log-idx');
      log.style.display = 'block';
      log.textContent = 'Error al construir el índice.';
    }).finally(function(){
      document.getElementById('load-idx').classList.remove('show');
      ids.forEach(function(id){document.getElementById(id).disabled = false});
    });
}

function runCmp() {
  document.getElementById('btn-run-cmp').disabled = true;
  document.getElementById('load-cmp').classList.add('show');
  document.getElementById('log-cmp').style.display = 'none';

  fetch('/run_comparison', {method:'POST'})
    .then(function(r){return r.json()}).then(function(data){
      var log = document.getElementById('log-cmp');
      log.style.display = 'block';
      log.innerHTML = data.log.replace(/\n/g,'<br>');
      log.scrollTop = log.scrollHeight;
      if (data.rag_rouge1 !== undefined) {
        document.getElementById('cmp-rag-r1').textContent = data.rag_rouge1.toFixed(3);
        document.getElementById('cmp-bl-r1').textContent = data.bl_rouge1.toFixed(3);
      }
      compData = [];
    }).catch(function(){
      var log = document.getElementById('log-cmp');
      log.style.display = 'block';
      log.textContent = 'Error al ejecutar la comparación.';
    }).finally(function(){
      document.getElementById('load-cmp').classList.remove('show');
      document.getElementById('btn-run-cmp').disabled = false;
    });
}

document.getElementById('query').addEventListener('keydown', function(e){
  if (e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    doAsk();
  }
});
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query', '').strip()
    mode = data.get('mode', 'rag')

    if not query:
        return jsonify({'error': 'Pregunta vacía'}), 400

    result = {'rag_answer': '', 'baseline_answer': '', 'rag_sources': [], 'agent_steps': []}

    try:
        if mode == 'agente':
            from src.agent.rag_agent import run_agent
            agent_result = run_agent(query)
            result['rag_answer'] = agent_result['answer']
            result['rag_sources'] = agent_result['sources']
            result['agent_steps'] = agent_result['steps']
            save_live_metrics(query, mode, result)
            return jsonify(result)

        vs = get_vector_store()

        if mode in ('rag', 'ambos'):
            from src.evaluation.rag_pipeline import answer_with_rag_multimodal
            rag_answer, rag_sources, retrieved_docs = answer_with_rag_multimodal(query, vs, k=4)
            result['rag_answer'] = rag_answer
            result['rag_sources'] = deduplicate_sources(rag_sources)
            result['retrieved_docs'] = retrieved_docs

        if mode in ('baseline', 'ambos'):
            from src.evaluation.baseline import answer_without_rag
            result['baseline_answer'] = answer_without_rag(query)

        save_live_metrics(query, mode, result)

        result.pop("retrieved_docs", None)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    try:
        live_path = Path("data/live_metrics.json")
        ragas_path = Path(EXPERIMENTS_DIR) / "ragas_results.json"

        response = {
            "items": [],
            "ragas": {}
        }

        if live_path.exists():
            with open(live_path, "r", encoding="utf-8") as f:
                response["items"] = json.load(f)

        if ragas_path.exists():
            with open(ragas_path, "r", encoding="utf-8") as f:
                ragas_data = json.load(f)

            rows = ragas_data.get("rows", [])
            averages = ragas_data.get("averages", {})

            # Si hay averages precalculados, usarlos directamente
            if averages:
                response["ragas"] = {
                    "n_questions": len(rows),
                    "faithfulness": averages.get("faithfulness", 0),
                    "answer_relevancy": averages.get("answer_relevancy", 0),
                    "context_precision": averages.get("context_precision", 0),
                    "context_recall": averages.get("context_recall", 0),
                }
            elif rows:
                # Fallback: calcular promedios desde rows (formato antiguo)
                def avg_metric(metric_name):
                    values = [
                        row.get(metric_name)
                        for row in rows
                        if isinstance(row.get(metric_name), (int, float))
                    ]
                    return round(sum(values) / len(values), 4) if values else 0

                response["ragas"] = {
                    "n_questions": len(rows),
                    "faithfulness": avg_metric("faithfulness"),
                    "answer_relevancy": avg_metric("answer_relevancy"),
                    "context_precision": avg_metric("context_precision"),
                    "context_recall": avg_metric("context_recall"),
                }

        if not response["items"] and not response["ragas"].get("n_questions"):
            return jsonify({"error": "No hay métricas todavía"}), 404

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/comparison')
def comparison():
    try:
        with open(Path(EXPERIMENTS_DIR) / 'rag_results.json', encoding='utf-8') as f:
            rag = json.load(f)

        with open(Path(EXPERIMENTS_DIR) / 'baseline_results.json', encoding='utf-8') as f:
            bl = json.load(f)

        bl_map = {r['id']: r for r in bl}
        combined = []

        for r in rag:
            b = bl_map.get(r['id'], {})
            combined.append({
                'id': r['id'],
                'question': r['question'],
                'category': r.get('category', ''),
                'rag_answer': r.get('answer', ''),
                'rag_sources': deduplicate_sources(r.get('sources', [])),
                'rag_metrics': r.get('metrics', {}),
                'baseline_answer': b.get('answer', ''),
                'baseline_metrics': b.get('metrics', {}),
            })

        return jsonify(combined)

    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/index_status')
def index_status():
    chroma_p = Path(VECTOR_STORE_PATH) / 'chroma.sqlite3'
    exists = chroma_p.exists()
    result = {'exists': exists}

    if exists:
        mtime = chroma_p.stat().st_mtime
        result['modified'] = time.strftime('%d/%m/%Y %H:%M', time.localtime(mtime))
        size_mb = round(sum(f.stat().st_size for f in Path(VECTOR_STORE_PATH).rglob('*') if f.is_file()) / 1024 / 1024, 1)
        result['type'] = f'ChromaDB ({size_mb} MB)'

    return jsonify(result)


@app.route('/build_index', methods=['POST'])
def build_index_route():
    mode = request.json.get('mode', 'texto')
    log = io.StringIO()

    try:
        with redirect_stdout(log):
            from src.ingestion.pdf_loader import load_pdfs_from_folder
            from src.ingestion.chunking import split_documents

            print("Cargando PDFs...")
            docs = load_pdfs_from_folder(CONFIG["paths"]["raw_data"])
            print(f"Páginas cargadas: {len(docs)}")
            chunks = split_documents(docs)

            if mode == 'multimodal':
                from src.multimodal.indexer import build_combined_index
                print("Construyendo índice multimodal...")
                build_combined_index(chunks, CONFIG["paths"]["raw_data"], VECTOR_STORE_PATH)
            else:
                from src.embeddings.vector_store import build_vector_store
                print("Construyendo índice de texto...")
                build_vector_store(chunks, VECTOR_STORE_PATH)

            print("Índice creado correctamente.")

            global _vector_store
            _vector_store = None

    except Exception as e:
        log.write(f"\nERROR: {e}")

    return jsonify({'log': log.getvalue()})


@app.route('/add_document', methods=['POST'])
def add_document_route():
    """
    Recibe un PDF por upload, lo procesa en chunks y lo añade
    incrementalmente a la BD vectorial sin reconstruir el índice.
    """
    log = io.StringIO()

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se ha enviado ningún archivo.'}), 400

        file = request.files['file']

        if not file.filename or not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Solo se aceptan archivos PDF.'}), 400

        with redirect_stdout(log):
            from src.ingestion.chunking import split_documents
            from src.embeddings.vector_store import add_documents

            # Guardar el PDF en data/raw
            raw_path = Path(CONFIG["paths"]["raw_data"])
            raw_path.mkdir(parents=True, exist_ok=True)

            save_path = raw_path / file.filename
            file.save(str(save_path))
            print(f"PDF guardado: {file.filename}")

            # Cargar y dividir en chunks
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(save_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file.filename

            print(f"Páginas cargadas: {len(docs)}")

            chunks = split_documents(docs)
            print(f"Chunks generados: {len(chunks)}")

            # Añadir incrementalmente
            stats = add_documents(chunks, VECTOR_STORE_PATH)
            print(f"Añadidos: {stats['añadidos']} | "
                  f"Duplicados ignorados: {stats['duplicados']} | "
                  f"Total recibidos: {stats['total_recibidos']}")

            # Invalidar cache del vector store
            global _vector_store
            _vector_store = None

            print("Documento añadido al índice correctamente.")

    except Exception as e:
        log.write(f"\nERROR: {e}")
        return jsonify({'error': str(e), 'log': log.getvalue()}), 500

    return jsonify({'log': log.getvalue(), 'filename': file.filename})


@app.route('/indexed_sources')
def indexed_sources_route():
    """Devuelve la lista de documentos indexados en la BD vectorial."""
    try:
        from src.embeddings.vector_store import list_indexed_sources
        sources = list_indexed_sources(VECTOR_STORE_PATH)

        total_chunks = sum(s["chunk_count"] for s in sources)

        return jsonify({
            'sources': sources,
            'total_documents': len(sources),
            'total_chunks': total_chunks,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_source', methods=['POST'])
def delete_source_route():
    """Elimina todos los chunks de un documento concreto de la BD."""
    data = request.json
    source_file = data.get('source_file', '').strip()

    if not source_file:
        return jsonify({'error': 'No se ha indicado el documento a eliminar.'}), 400

    try:
        from src.embeddings.vector_store import delete_source

        deleted = delete_source(source_file, VECTOR_STORE_PATH)

        global _vector_store
        _vector_store = None

        return jsonify({
            'deleted_chunks': deleted,
            'source_file': source_file,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/run_comparison', methods=['POST'])
def run_comparison_route():
    log = io.StringIO()
    rag_rouge1 = 0.0
    bl_rouge1 = 0.0

    try:
        with redirect_stdout(log):
            from src.evaluation.dataset_builder import build_default_dataset    
            DEFAULT_QUESTIONS = build_default_dataset()
            from src.evaluation.rag_pipeline import answer_with_rag_multimodal
            from src.evaluation.baseline import answer_without_rag
            from src.evaluation.metrics import compute_all_metrics
            from src.evaluation.ragas_eval import run_ragas_eval, format_ragas_result

            vs = get_vector_store()
            rag_res, bl_res = [], []

            for item in DEFAULT_QUESTIONS[:5]:
                q = item['question']
                ref = item.get('reference_answer')

                print(f"[{item['id']}] {q[:55]}...")

                ra, rs, rd = answer_with_rag_multimodal(q, vs, k=4)
                ba = answer_without_rag(q)

                rs = deduplicate_sources(rs)

                print(f"  RAG: {ra[:70]}...")

                rm = compute_all_metrics(
                    ra,
                    rag_sources=rs,
                    retrieved_docs=rd,
                    reference_answer=ref
                )

                bm = compute_all_metrics(
                    ba,
                    reference_answer=ref
                )

                rag_res.append({
                    'id': item['id'],
                    'question': q,
                    'category': item.get('category', ''),
                    'answer': ra,
                    'sources': rs,
                    'retrieved_docs': [
                        doc.page_content if hasattr(doc, "page_content") else str(doc)
                        for doc in rd
                    ],
                    'reference_answer': ref,
                    'metrics': rm
                })

                bl_res.append({
                    'id': item['id'],
                    'question': q,
                    'category': item.get('category', ''),
                    'answer': ba,
                    'reference_answer': ref,
                    'metrics': bm
                })

            exp = Path(EXPERIMENTS_DIR)
            exp.mkdir(exist_ok=True)

            with open(exp / 'rag_results.json', 'w', encoding='utf-8') as f:
                json.dump(rag_res, f, indent=2, ensure_ascii=False)

            with open(exp / 'baseline_results.json', 'w', encoding='utf-8') as f:
                json.dump(bl_res, f, indent=2, ensure_ascii=False)

            def avg(lst, k):
                vals = [
                    r['metrics'].get(k)
                    for r in lst
                    if r.get('metrics', {}).get(k) is not None
                ]
                return round(sum(vals) / len(vals), 4) if vals else 0

            keys = [
                'answer_length_words',
                'faithfulness',
                'rouge1_vs_context',
                'rougeL_vs_context',
                'rouge1_f1',
                'rouge2_f1',
                'rougeL_f1',
                'is_empty',
                'mentions_no_info'
            ]

            summary = [
                {
                    'system': 'RAG',
                    'n_questions': len(rag_res),
                    **{f'avg_{k}': avg(rag_res, k) for k in keys}
                },
                {
                    'system': 'Baseline',
                    'n_questions': len(bl_res),
                    **{f'avg_{k}': avg(bl_res, k) for k in keys}
                },
            ]

            with open(exp / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            rag_rouge1 = avg(rag_res, 'rouge1_f1')
            bl_rouge1 = avg(bl_res, 'rouge1_f1')

            print(f"\nRAG rouge1={rag_rouge1:.3f} | Baseline rouge1={bl_rouge1:.3f}")
            print("Resultados clásicos guardados en experiments/")

            # ==============================
            # RAGAS
            # ==============================
            print("\nEjecutando evaluación RAGAS...")
            print("Métricas: faithfulness, context_precision, answer_relevancy, context_recall")

            ragas_questions = []
            ragas_answers = []
            ragas_contexts = []
            ragas_references = []

            for r in rag_res:
                ragas_questions.append(r["question"])
                ragas_answers.append(r["answer"])

                contexts = []

                if r.get("retrieved_docs"):
                    contexts = r["retrieved_docs"]
                elif r.get("sources"):
                    contexts = [
                        s.get("content_preview", "")
                        for s in r["sources"]
                        if s.get("content_preview")
                    ]

                # Asegurar que contexts no esté vacío (RAGAS lo requiere)
                if not contexts:
                    contexts = [""]

                ragas_contexts.append(contexts)
                ragas_references.append(r.get("reference_answer") or "")

            try:
                print("Usando Ollama directamente para evaluación RAGAS...")

                ragas_result = run_ragas_eval(
                    questions=ragas_questions,
                    answers=ragas_answers,
                    contexts=ragas_contexts,
                    references=ragas_references,
                )

                ragas_dict = format_ragas_result(ragas_result)

                with open(exp / "ragas_results.json", "w", encoding="utf-8") as f:
                    json.dump(ragas_dict, f, indent=2, ensure_ascii=False)

                # Mostrar resumen en el log
                avgs = ragas_dict.get("averages", {})
                if avgs:
                    print("\nResultados RAGAS (promedios):")
                    for metric, value in avgs.items():
                        print(f"  {metric}: {value:.4f}")

                print("RAGAS guardado en experiments/ragas_results.json")

            except Exception as ragas_error:
                print(f"ERROR en RAGAS: {ragas_error}")
                import traceback
                traceback.print_exc()
                print("La comparación clásica se ha guardado correctamente igualmente.")

    except Exception as e:
        log.write(f"\nERROR: {e}")

    return jsonify({
        'log': log.getvalue(),
        'rag_rouge1': rag_rouge1,
        'bl_rouge1': bl_rouge1
    })

if __name__ == '__main__':
    print("\nIniciando interfaz web...")
    print("Abre tu navegador en: http://localhost:5001\n")
    app.run(debug=False, port=5001)
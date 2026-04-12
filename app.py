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
.answer{font-size:.95rem;line-height:1.7}
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

<!-- CONSULTA -->
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

<!-- COMPARACION -->
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

<!-- METRICAS -->
<div id="panel-metricas" class="panel">
  <div id="res-metrics"><div class="empty">Cargando métricas...</div></div>
</div>

<!-- INDICE -->
<div id="panel-indice" class="panel">
  <div class="section-title">Gestión del índice</div>
  <div id="indice-status" style="margin-bottom:2rem"></div>

  <div class="grid2" style="margin-bottom:2rem">
    <div class="card">
      <div class="card-head"><span class="badge badge-rag">Solo texto</span></div>
      <div class="card-body">
        <div class="answer" style="font-size:.88rem;margin-bottom:1rem">Indexa únicamente chunks de texto. Más rápido y con mejores métricas textuales.</div>
        <button class="btn btn-primary" id="btn-idx-texto" onclick="buildIdx('texto')">Construir índice de texto</button>
      </div>
    </div>
    <div class="card">
      <div class="card-head"><span class="badge badge-accent">Texto + Imágenes</span></div>
      <div class="card-body">
        <div class="answer" style="font-size:.88rem;margin-bottom:1rem">Indexa texto + captions de imágenes con LLaVA. Requiere Ollama con LLaVA instalado.</div>
        <button class="btn btn-secondary" id="btn-idx-multi" onclick="buildIdx('multimodal')">Construir índice multimodal</button>
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
  if (curMode === 'ambos') {
    cont.innerHTML = buildGrid(data.rag_answer, data.rag_sources, data.baseline_answer, null);
    return;
  }
  var isRag = curMode === 'rag';
  var ans = isRag ? data.rag_answer : data.baseline_answer;
  var src = isRag ? (data.rag_sources || []) : [];
  var badgeCls = isRag ? 'badge-rag' : 'badge-bl';
  var lbl = isRag ? 'RAG' : 'BASELINE';
  cont.innerHTML = '<div class="card anim"><div class="card-head"><span class="badge ' + badgeCls + '">' + lbl + '</span></div><div class="card-body"><div class="answer">' + ans + '</div>' + srcHtml(src) + '</div></div>';
}

function buildGrid(ragAns, ragSrc, blAns, blSrc) {
  return '<div class="grid2">' +
    '<div class="card anim"><div class="card-head"><span class="badge badge-rag">RAG</span></div><div class="card-body"><div class="answer">' + ragAns + '</div>' + srcHtml(ragSrc || []) + '</div></div>' +
    '<div class="card anim"><div class="card-head"><span class="badge badge-bl">Baseline</span></div><div class="card-body"><div class="answer">' + blAns + '</div></div></div>' +
  '</div>';
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
    renderMetrics(data.rag, data.baseline);
  }).catch(function(){
    document.getElementById('res-metrics').innerHTML = '<div class="empty">No hay métricas. Ejecuta la comparación primero.</div>';
  });
}

function renderMetrics(rag, bl) {
  var keys = [
    {k:'avg_rouge1_f1', lbl:'ROUGE-1 F1'},
    {k:'avg_rouge2_f1', lbl:'ROUGE-2 F1'},
    {k:'avg_rougeL_f1', lbl:'ROUGE-L F1'},
    {k:'avg_mentions_no_info', lbl:'Sin respuesta'},
    {k:'avg_answer_length_words', lbl:'Longitud media'}
  ];

  var cards = ['avg_rouge1_f1','avg_rouge2_f1','avg_rougeL_f1'].map(function(k){
    var rv = rag[k] || 0;
    var bv = bl[k] || 0;
    var better = rv >= bv;
    return '<div class="metric-card">' +
      '<div class="label">' + k.replace('avg_','').replace(/_/g,' ').toUpperCase() + '</div>' +
      '<div class="metric-val" style="color:var(--rag)">' + rv.toFixed(3) + '</div>' +
      '<div class="metric-cmp ' + (better?'better':'worse') + '">' + (better?'▲':'▼') + ' vs baseline ' + bv.toFixed(3) + '</div>' +
    '</div>';
  }).join('');

  var noInfoR = ((rag.avg_mentions_no_info || 0) * 100).toFixed(0);
  var noInfoB = ((bl.avg_mentions_no_info || 0) * 100).toFixed(0);
  var noInfoBetter = (rag.avg_mentions_no_info || 0) <= (bl.avg_mentions_no_info || 0);

  cards += '<div class="metric-card">' +
    '<div class="label">SIN RESPUESTA</div>' +
    '<div class="metric-val" style="color:' + (noInfoBetter ? 'var(--accent2)' : 'var(--accent3)') + '">' + noInfoR + '%</div>' +
    '<div class="metric-cmp ' + (noInfoBetter ? 'better' : 'worse') + '">Baseline: ' + noInfoB + '%</div>' +
  '</div>';

  var bars = keys.map(function(m){
    var rv = rag[m.k] || 0;
    var bv = bl[m.k] || 0;
    var mx = Math.max(rv, bv, 0.001);
    var fmt = function(v){ return v < 2 ? v.toFixed(3) : Math.round(v); };
    return '<div style="margin-bottom:1.5rem">' +
      '<div class="label">' + m.lbl + '</div>' +
      '<div class="bar-row"><div class="bar-lbl">RAG</div><div class="bar-track"><div class="bar-fill rag" style="width:' + (rv/mx*100).toFixed(1) + '%"></div></div><div class="bar-val">' + fmt(rv) + '</div></div>' +
      '<div class="bar-row"><div class="bar-lbl">Baseline</div><div class="bar-track"><div class="bar-fill bl" style="width:' + (bv/mx*100).toFixed(1) + '%"></div></div><div class="bar-val">' + fmt(bv) + '</div></div>' +
    '</div>';
  }).join('');

  document.getElementById('res-metrics').innerHTML =
    '<div class="section-title">Resultados de la evaluación</div>' +
    '<div style="display:flex;gap:1.5rem;margin-bottom:1.5rem">' +
      '<span class="mono small dim"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--rag);margin-right:.4rem"></span>RAG (' + rag.n_questions + ' preguntas)</span>' +
      '<span class="mono small dim"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--bl);margin-right:.4rem"></span>Baseline (' + bl.n_questions + ' preguntas)</span>' +
    '</div>' +
    '<div class="grid2" style="margin-bottom:2rem">' + cards + '</div>' +
    '<div>' + bars + '</div>';
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

    result = {'rag_answer': '', 'baseline_answer': '', 'rag_sources': []}

    try:
        vs = get_vector_store()

        if mode in ('rag', 'ambos'):
            from src.evaluation.rag_pipeline import answer_with_rag_multimodal
            rag_answer, rag_sources, _ = answer_with_rag_multimodal(query, vs, k=4)
            result['rag_answer'] = rag_answer
            result['rag_sources'] = deduplicate_sources(rag_sources)

        if mode in ('baseline', 'ambos'):
            from src.evaluation.baseline import answer_without_rag
            result['baseline_answer'] = answer_without_rag(query)

    except Exception as e:
        result['error'] = str(e)

    return jsonify(result)


@app.route('/metrics')
def metrics():
    try:
        with open(Path(EXPERIMENTS_DIR) / 'summary.json', encoding='utf-8') as f:
            summary = json.load(f)

        rag = next((s for s in summary if s['system'] == 'RAG'), {})
        bl = next((s for s in summary if s['system'] == 'Baseline'), {})
        return jsonify({'rag': rag, 'baseline': bl})

    except Exception as e:
        return jsonify({'error': str(e)}), 404


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
    faiss_p = Path(VECTOR_STORE_PATH) / 'index.faiss'
    exists = faiss_p.exists()
    result = {'exists': exists}

    if exists:
        mtime = faiss_p.stat().st_mtime
        result['modified'] = time.strftime('%d/%m/%Y %H:%M', time.localtime(mtime))
        size_mb = round(faiss_p.stat().st_size / 1024 / 1024, 1)
        result['type'] = f'FAISS ({size_mb} MB)'

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


@app.route('/run_comparison', methods=['POST'])
def run_comparison_route():
    log = io.StringIO()
    rag_rouge1 = 0.0
    bl_rouge1 = 0.0

    try:
        with redirect_stdout(log):
            from src.evaluation.dataset_builder import DEFAULT_QUESTIONS
            from src.evaluation.rag_pipeline import answer_with_rag_multimodal
            from src.evaluation.baseline import answer_without_rag
            from src.evaluation.metrics import compute_all_metrics

            vs = get_vector_store()
            rag_res, bl_res = [], []

            for item in DEFAULT_QUESTIONS:
                q = item['question']
                ref = item.get('reference_answer')

                print(f"[{item['id']}] {q[:55]}...")
                ra, rs, rd = answer_with_rag_multimodal(q, vs, k=4)
                ba = answer_without_rag(q)

                rs = deduplicate_sources(rs)

                print(f"  RAG: {ra[:70]}...")
                rm = compute_all_metrics(ra, rag_sources=rs, retrieved_docs=rd, reference_answer=ref)
                bm = compute_all_metrics(ba, reference_answer=ref)

                rag_res.append({
                    'id': item['id'],
                    'question': q,
                    'category': item.get('category', ''),
                    'answer': ra,
                    'sources': rs,
                    'metrics': rm
                })

                bl_res.append({
                    'id': item['id'],
                    'question': q,
                    'category': item.get('category', ''),
                    'answer': ba,
                    'metrics': bm
                })

            exp = Path(EXPERIMENTS_DIR)
            exp.mkdir(exist_ok=True)

            with open(exp / 'rag_results.json', 'w', encoding='utf-8') as f:
                json.dump(rag_res, f, indent=2, ensure_ascii=False)

            with open(exp / 'baseline_results.json', 'w', encoding='utf-8') as f:
                json.dump(bl_res, f, indent=2, ensure_ascii=False)

            def avg(lst, k):
                vals = [r['metrics'].get(k) for r in lst if r['metrics'].get(k) is not None]
                return round(sum(vals) / len(vals), 4) if vals else 0

            keys = [
                'answer_length_words', 'faithfulness', 'rouge1_vs_context',
                'rougeL_vs_context', 'rouge1_f1', 'rouge2_f1',
                'rougeL_f1', 'is_empty', 'mentions_no_info'
            ]

            summary = [
                {'system': 'RAG', 'n_questions': len(rag_res), **{f'avg_{k}': avg(rag_res, k) for k in keys}},
                {'system': 'Baseline', 'n_questions': len(bl_res), **{f'avg_{k}': avg(bl_res, k) for k in keys}},
            ]

            with open(exp / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            rag_rouge1 = avg(rag_res, 'rouge1_f1')
            bl_rouge1 = avg(bl_res, 'rouge1_f1')

            print(f"\nRAG rouge1={rag_rouge1:.3f} | Baseline rouge1={bl_rouge1:.3f}")
            print("Guardado en experiments/")

    except Exception as e:
        log.write(f"\nERROR: {e}")

    return jsonify({'log': log.getvalue(), 'rag_rouge1': rag_rouge1, 'bl_rouge1': bl_rouge1})


if __name__ == '__main__':
    print("\n Iniciando interfaz web...")
    print("   Abre tu navegador en: http://localhost:5001\n")
    app.run(debug=False, port=5001)
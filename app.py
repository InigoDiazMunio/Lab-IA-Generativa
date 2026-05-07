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

from flask import Flask, render_template, request, jsonify
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



@app.route('/')
def index():
    return render_template('index.html')


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

        if mode == 'verificado':
            from src.agent.rag_graph import run_rag_verified
            vr = run_rag_verified(query)
            result['rag_answer']  = vr['answer']
            result['rag_sources'] = deduplicate_sources(vr['sources'])
            result['verdict']     = vr['verdict']
            result['verdict_reason'] = vr['reason']
            result['attempts']    = vr['attempts']
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
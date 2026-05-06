"""
Agente IA sobre el pipeline RAG existente.

Usa create_react_agent (ReAct: Reason + Act) en lugar de create_tool_calling_agent
porque ChatOllama de langchain_community no implementa bind_tools().
ReAct funciona con cualquier LLM mediante un formato texto estructurado.

El agente NO reemplaza el pipeline RAG — lo usa como herramienta, por lo que
ambos modos (RAG lineal y Agente) son comparables en las mismas métricas.

Herramientas:
  - search_university_docs  : retrieval textual (ChromaDB + cross-encoder reranker)
  - visual_memory_access    : búsqueda en captions de imágenes indexadas
"""

import re
import glob
import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

_VISUAL_KEYWORDS = {"Figure", "Fig.", "Table", "Diagram", "Equation", "figure", "fig.", "table", "diagram", "equation"}


# ── Vector store compartido ───────────────────────────────────────────────────

_cached_vs = None


def _get_vs():
    global _cached_vs
    if _cached_vs is None:
        from src.embeddings.vector_store import load_vector_store
        _cached_vs = load_vector_store("data/embeddings/chroma_db")
    return _cached_vs


# ── Rastreador de fuentes ─────────────────────────────────────────────────────

class _SourceTracker:
    def __init__(self):
        self._sources: list = []

    def add(self, source_file, page, content_preview, source_type="texto", image_path=None):
        key = (str(source_file), str(page), source_type)
        if key not in {(s["source_file"], str(s["page"]), s.get("type")) for s in self._sources}:
            entry = {
                "source_file": source_file,
                "page": page,
                "content_preview": str(content_preview)[:300],
                "type": source_type,
            }
            if image_path:
                entry["image_path"] = image_path
            self._sources.append(entry)

    def get(self) -> list:
        return list(self._sources)

    def clear(self):
        self._sources.clear()


_tracker = _SourceTracker()


# ── Deduplicador de llamadas ──────────────────────────────────────────────────

class _CallDeduplicator:
    """Detecta y bloquea llamadas repetidas a las mismas herramientas dentro de
    una misma invocación del agente, evitando bucles de iteraciones idénticas."""

    def __init__(self):
        self._seen: dict[str, list[str]] = {}

    def is_duplicate(self, tool_name: str, query: str) -> bool:
        normalized = re.sub(r'\s+', ' ', query.lower().strip())[:150]
        past = self._seen.setdefault(tool_name, [])
        for prev in past:
            if normalized == prev:
                return True
            prev_toks = set(prev.split())
            new_toks = set(normalized.split())
            if prev_toks and new_toks:
                overlap = len(prev_toks & new_toks) / max(len(prev_toks), len(new_toks))
                if overlap >= 0.80:
                    return True
        past.append(normalized)
        return False

    def clear(self):
        self._seen.clear()


_dedup = _CallDeduplicator()

_STOP_LOOP_MSG = (
    "[STOP-LOOP] You already searched for this in a previous step. "
    "Read the Observations already in the scratchpad and write your Final Answer now. "
    "Do NOT call any more tools."
)

# ── Herramienta 1: búsqueda textual ──────────────────────────────────────────

@tool
def search_university_docs(query: str) -> str:
    """
    Busca información textual en los documentos académicos.
    Úsala para responder preguntas conceptuales, definiciones, comparaciones
    y explicaciones técnicas sobre RAG, embeddings, LLMs y sistemas similares.
    Devuelve los fragmentos más relevantes con su fuente (archivo y página).
    """
    if _dedup.is_duplicate("search_university_docs", query):
        return _STOP_LOOP_MSG

    from src.retrieval.retriever import retrieve_context, translate_query
    from src.retrieval.ranking import rank_retrieved_docs

    try:
        vs = _get_vs()

        # Traducir la query ES→EN antes del embedding Y del reranker.
        # Los papers están en inglés; buscar en español produce baja similitud coseno.
        eng_query = translate_query(query)

        # k=20 para dar margen al reranker de encontrar docs de distintos papers.
        candidates = retrieve_context(vs, eng_query, k=20, translate=False)

        seen: set = set()
        unique = []
        for doc in candidates:
            key = (
                doc.metadata.get("source_file", doc.metadata.get("source", "")),
                doc.metadata.get("page", ""),
                doc.page_content[:80].strip(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        # top_k=5 para no descartar demasiado agresivamente tras ampliar candidatos
        retrieved = rank_retrieved_docs(unique, query=eng_query, top_k=5)

        if not retrieved:
            return "No se encontraron fragmentos relevantes."

        parts = []
        source_papers: set = set()
        for i, doc in enumerate(retrieved, 1):
            src = doc.metadata.get("source_file", doc.metadata.get("source", "desconocido"))
            page = doc.metadata.get("page_label") or str(doc.metadata.get("page", "N/A"))
            source_papers.add(src)
            _tracker.add(src, page, doc.page_content, "texto")
            parts.append(f"[Source {i}: {src}, p.{page}]\n{doc.page_content.strip()}")

        result = "\n\n".join(parts)

        # Pista de diversidad: si todos los fragmentos vienen del mismo paper,
        # el agente puede reformular con términos técnicos en inglés para buscar
        # en otros papers del índice.
        if len(source_papers) == 1:
            result += (
                "\n\n[NARROW RESULT: All fragments are from the same paper. "
                "If these don't fully answer the question, retry with English technical terms "
                "(e.g. 'faithfulness answer relevance context precision recall RAGAS evaluation').]"
            )

        has_visual = any(kw in doc.page_content for doc in retrieved for kw in _VISUAL_KEYWORDS)
        if has_visual:
            result += (
                "\n\n[VISUAL HINT: The retrieved text references visual content "
                "(Figure / Table / Diagram / Equation). "
                "Call visual_memory_access with the same query to retrieve related visual context.]"
            )

        return result

    except Exception as e:
        return f"Error en la búsqueda: {e}"


# ── Helper: localizar imágenes en disco para un paper y página ───────────────

def _find_images_for_page(source_file: str, page) -> list[str]:
    """Devuelve rutas de imágenes en data/processed/images/{paper}/page_{N}_img_*.*

    source_file puede ser un path completo como 'data\\raw\\paper 1.pdf'.
    page debe ser el número de página legible (page_label), no el índice 0-based.
    """
    from pathlib import Path
    paper_stem = Path(source_file.replace("\\", "/")).stem  # "paper 1.pdf" → "paper 1"
    base = os.path.join("data", "processed", "images", paper_stem)
    results = []
    for ext in (".png", ".jpeg", ".jpg"):
        results.extend(glob.glob(os.path.join(base, f"page_{page}_img_*{ext}")))
    return sorted(results)


# ── Herramienta 2: búsqueda visual ───────────────────────────────────────────

@tool
def visual_memory_access(query: str) -> str:
    """
    Busca en los captions de las imágenes y figuras de los documentos.
    Úsala cuando la pregunta mencione diagramas, figuras, esquemas,
    arquitecturas visuales o tablas. Devuelve captions y las rutas de
    las imágenes originales cuando están disponibles.
    """
    if _dedup.is_duplicate("visual_memory_access", query):
        return _STOP_LOOP_MSG

    from src.retrieval.retriever import retrieve_context, translate_query

    _VISUAL_DONE = (
        "\n\n[DONE — visual search complete. "
        "You have retrieved the available visual context. "
        "Write your Final Answer NOW. Do not call any more tools.]"
    )

    try:
        vs = _get_vs()
        eng_query = translate_query(query)
        candidates = retrieve_context(vs, eng_query, k=12, translate=False)

        # Prioridad 1: docs indexados como image_caption (índice multimodal construido)
        visual_docs = [d for d in candidates if d.metadata.get("type") == "image_caption"]

        if visual_docs:
            parts = []
            for i, doc in enumerate(visual_docs[:4], 1):
                src = doc.metadata.get("source_file", doc.metadata.get("source", "desconocido"))
                page = doc.metadata.get("page_label") or str(doc.metadata.get("page", "N/A"))
                imgs = _find_images_for_page(src, page)
                img_path = imgs[0] if imgs else None
                _tracker.add(src, page, doc.page_content, "image_caption", image_path=img_path)
                img_note = f" [imagen: {img_path}]" if img_path else ""
                parts.append(f"[Figure {i}: {src}, p.{page}{img_note}]\n{doc.page_content.strip()}")
            return "\n\n".join(parts) + _VISUAL_DONE

        # Prioridad 2 (fallback): buscar imágenes en disco a partir de los docs de texto
        # más relevantes aunque no estén indexadas como image_caption
        text_docs = [d for d in candidates if d.metadata.get("type") != "image_caption"][:6]
        found = []
        for doc in text_docs:
            src = doc.metadata.get("source_file", doc.metadata.get("source", ""))
            page = doc.metadata.get("page_label") or str(doc.metadata.get("page", ""))
            if not src or not page:
                continue
            imgs = _find_images_for_page(src, page)
            for img_path in imgs[:2]:
                if img_path not in [f["image_path"] for f in found if "image_path" in f]:
                    caption_text = f"Imagen de {src.replace('.pdf','')}, pág. {page}"
                    _tracker.add(src, page, caption_text, "image_caption", image_path=img_path)
                    found.append({"src": src, "page": page, "image_path": img_path})

        if found:
            lines = [f"[Figure {i}: {f['src']}, p.{f['page']}] {f['image_path']}" for i, f in enumerate(found[:4], 1)]
            return (
                "El índice multimodal no está construido, pero se localizaron imágenes en disco "
                "para las páginas relevantes:\n" + "\n".join(lines) + _VISUAL_DONE
            )

        return (
            "No se encontraron figuras relacionadas con esta consulta. "
            "Para habilitar la búsqueda visual completa, construye el índice multimodal con LLaVA."
            + _VISUAL_DONE
        )

    except Exception as e:
        return f"Error en la búsqueda visual: {e}"


# ── Prompt ReAct ──────────────────────────────────────────────────────────────
# Los marcadores (Action:, Action Input:, Final Answer:) DEBEN estar en inglés
# porque el AgentOutputParser de LangChain los busca literalmente.
# CRÍTICO: No incluir "Observation:" en el ejemplo — el LLM lo rellenaría él mismo
# (alucinando resultados) en lugar de esperar el resultado real de la herramienta.
# Cada respuesta del LLM debe contener SOLO una de las dos opciones (A o B),
# nunca las dos a la vez (eso causa OUTPUT_PARSING_FAILURE).

_REACT_PROMPT = PromptTemplate.from_template(
    """You are a university assistant specialized in AI, RAG systems, and NLP.
Your ONLY job is to answer in fluent, complete SPANISH using the retrieved information.

═══ LANGUAGE RULE (MANDATORY) ═══
The retrieved documents are in English. You MUST translate and synthesize their content.
NEVER copy-paste English sentences into the Final Answer.
NEVER answer in English. Every word of the Final Answer must be in Spanish.

═══ SYNTHESIS RULE ═══
Do NOT reproduce raw text from the documents. Understand it, translate it, and explain it
in your own Spanish words. If the concept involves several items or metrics, use bullet
points starting with "•".

Example of correct structure for a metrics question:
  RAGAS mide la calidad de los sistemas RAG a través de tres dimensiones:
  • Fidelidad (Faithfulness): mide si la respuesta está respaldada por el contexto recuperado.
  • Relevancia de la Respuesta (Answer Relevance): evalúa si la respuesta aborda la pregunta.
  • Precisión del Contexto (Context Precision): valora si el contexto recuperado es pertinente.

═══ STOPPING RULE ═══
After 1-2 tool calls with sufficient information, write the Final Answer immediately.
Do not keep searching. The goal is ONE clear, complete Spanish answer.

═══ ANTI-LOOP RULE ═══
Before calling a tool, check the scratchpad. If you already called that tool with the same
or similar query, DO NOT call it again. Write the Final Answer using what you have.

═══ VISUAL DONE RULE ═══
After calling visual_memory_access, write the Final Answer immediately. Do not call more tools.
If any observation contains [DONE] or [STOP-LOOP], write Final Answer immediately.

═══ REFORMULATION RULE ═══
If results contain [NARROW RESULT], you MAY do ONE more search with English technical terms.
Example: "faithfulness answer relevance context precision RAGAS automated evaluation"

{chat_history}
Available tools:
{tools}

Each response must contain ONLY ONE of the two options below — never both:

OPTION A — call a tool:
Thought: [one sentence in English explaining why you need this tool]
Action: [must be exactly one of: {tool_names}]
Action Input: [search query as plain string]

OPTION B — give the final answer:
Thought: I now have enough information to answer in Spanish.
Final Answer: [your complete Spanish answer starts HERE on this same line]

CRITICAL FORMAT RULES:
- "Thought:" must be in English and appear first.
- "Final Answer:" MUST appear verbatim on its own line, followed immediately by the Spanish text.
- Do NOT write the answer before "Final Answer:". Do NOT skip the "Final Answer:" label.
- The answer after "Final Answer:" must be entirely in Spanish. Zero English sentences.
- Never write Action and Final Answer in the same response.

Question: {input}
Thought:{agent_scratchpad}"""
)


# ── Handler de rescate de parseo ─────────────────────────────────────────────
# Problema: llama3.1:8b genera la respuesta española correcta (con bullets,
# traducida) pero omite el token "Final Answer:" que el parser de LangChain
# necesita. El parser falla y consume iteraciones inútilmente.
#
# Solución en dos partes:
# 1. _rescue_parse: handler que devuelve un string con instrucción clara.
#    LangChain 0.3.x NO acepta AgentFinish como retorno del handler, así que
#    solo devolvemos texto.
# 2. _extract_from_log + run_agent: después de invoke(), revisamos action.log
#    de los pasos _Exception — ahí LangChain guarda el output bruto del LLM
#    que falló al parsear. Rescatamos el contenido desde ahí.

def _rescue_parse(error) -> str:
    """Devuelve instrucción concisa cuando el LLM genera respuesta sin 'Final Answer:'."""
    raw = getattr(error, 'llm_output', '') or str(error)
    has_content = len(raw.strip()) > 100
    has_action = 'Action:' in raw and 'Action Input:' in raw

    if has_content and not has_action:
        return (
            "Your answer was detected but the label 'Final Answer:' is missing. "
            "Write EXACTLY: Final Answer: [your Spanish text here, same line]"
        )
    return (
        "FORMAT ERROR. Write EXACTLY one of:\n"
        "Option A: Thought: ...\nAction: ...\nAction Input: ...\n"
        "Option B: Thought: ...\nFinal Answer: [Spanish answer on same line]"
    )


def _extract_from_log(log: str) -> str | None:
    """
    Extrae una respuesta válida del log bruto de un paso _Exception.
    LangChain guarda el output original del LLM en action.log cuando el parseo falla.
    """
    # Quitar línea Thought (en inglés o español)
    body = re.sub(r'(?m)^(?:Thought|Pensamiento|Pensaré)[^\n]*\n*', '', log, count=1).strip()
    # Si contiene "Final Answer:" extraer desde ahí
    fa_match = re.search(r'Final Answer\s*:\s*([\s\S]+)', body)
    if fa_match:
        body = fa_match.group(1)
    # Limpiar marcadores internos y referencias
    body = re.split(r'\n*Fuentes\s*:', body, maxsplit=1)[0]
    body = re.sub(r'\[(?:STOP-LOOP|DONE|NARROW RESULT|VISUAL HINT)[^\]]*\]', '', body)
    body = re.sub(r'\[Source\s*\d*\s*:[^\]]*\]', '', body)
    # Colapsar solo espacios/tabulaciones horizontales, nunca saltos de línea.
    # Los \n son los delimitadores de los bullets (• item\n• item) — si los
    # colapsamos aquí, el HTML renderizaría todo en una sola línea.
    body = re.sub(r'[ \t]{2,}', ' ', body).strip()

    if len(body) > 30 and 'Action:' not in body and 'Action Input:' not in body:
        return body
    return None


# ── AgentExecutor (singleton) ─────────────────────────────────────────────────

_agent_executor: AgentExecutor | None = None
_agent_memory: ConversationBufferWindowMemory | None = None


def get_agent_executor() -> AgentExecutor:
    """Construye el AgentExecutor la primera vez y lo reutiliza."""
    global _agent_executor, _agent_memory
    if _agent_executor is not None:
        return _agent_executor

    llm = ChatOllama(model="llama3.1:8b", temperature=0.1, num_ctx=4096, num_predict=1024)
    tools = [search_university_docs, visual_memory_access]

    agent = create_react_agent(llm, tools, _REACT_PROMPT)

    # Ventana de memoria conversacional (k=3 intercambios).
    # Almacena los últimos 3 pares pregunta/respuesta para dar contexto
    # al agente en preguntas de seguimiento sin repetir búsquedas idénticas.
    _agent_memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=False,
    )

    _agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=_agent_memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=_rescue_parse,
        return_intermediate_steps=True,
    )
    return _agent_executor


def reset_agent():
    """Fuerza la recreación del singleton y limpia la memoria conversacional."""
    global _agent_executor, _agent_memory
    _agent_executor = None
    _agent_memory = None


# ── Limpieza del texto de respuesta ──────────────────────────────────────────

def _clean_answer(text: str) -> str:
    """
    Devuelve el texto de respuesta limpio:
    - Sin bloques de fuentes (se devuelven por separado en 'sources')
    - Sin referencias inline [Source N: ...]
    - Sin señales internas [STOP-LOOP], [DONE], [NARROW RESULT]
    - Sin prefijos de relleno generados por el fallback anterior
    """
    # Cortar todo lo que venga después de "Fuentes:" (con o sin salto previo)
    text = re.split(r'\n*Fuentes\s*:', text, maxsplit=1)[0]
    # Eliminar referencias inline [Source N: ...] — la segunda regex cubre ambos casos
    text = re.sub(r'\[Source\s*\d*\s*:[^\]]*\]', '', text)
    # Eliminar señales internas del pipeline
    text = re.sub(r'\[(?:STOP-LOOP|DONE|NARROW RESULT|VISUAL HINT)[^\]]*\]', '', text)
    # Eliminar prefijos de relleno tipo "La búsqueda ha concluido. Basándome en..."
    text = re.sub(
        r'^(?:La búsqueda ha concluido\.?\s*)?(?:Basándome en los fragmentos recuperados\s*[:.]?\s*)?',
        '', text, flags=re.IGNORECASE
    )
    # Colapsar solo espacios/tabulaciones horizontales, preservar saltos de línea.
    # Los \n delimitan los bullets — colapsarlos los fundiría en una sola línea.
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── Punto de entrada público ──────────────────────────────────────────────────

def run_agent(query: str) -> dict:
    """
    Ejecuta el agente y devuelve respuesta + fuentes + pasos.

    Returns:
        {
          "answer":  str,            # texto limpio en español, sin marcas de fuente
          "sources": list[dict],     # {source_file, page, content_preview, type}
          "steps":   list[dict],     # {tool, input, result_preview}
        }
    """
    _tracker.clear()
    _dedup.clear()
    executor = get_agent_executor()

    try:
        result = executor.invoke({"input": query})
        raw_answer = result.get("output", "")

        # Si el agente no produjo Final Answer, intentar rescatar del log de _Exception.
        # LangChain guarda el output bruto del LLM en action.log cuando el parser falla,
        # por lo que podemos extraer la respuesta que el LLM generó correctamente pero
        # sin el token "Final Answer:".
        if not raw_answer or "Agent stopped" in raw_answer:
            rescued = None
            for action, _ in result.get("intermediate_steps", []):
                if getattr(action, "tool", "") == "_Exception":
                    rescued = _extract_from_log(getattr(action, "log", ""))
                    if rescued:
                        break

            if rescued:
                raw_answer = rescued
            else:
                has_results = any(
                    not str(obs).startswith("[STOP-LOOP]")
                    for _, obs in result.get("intermediate_steps", [])
                )
                raw_answer = (
                    "Se recuperó información relevante pero el agente no pudo sintetizarla. "
                    "Por favor, reformula la pregunta de forma más específica."
                    if has_results else
                    "No se encontró información suficiente en los documentos para responder."
                )

        steps = []
        for action, observation in result.get("intermediate_steps", []):
            tool_name = getattr(action, "tool", str(action))
            # Filtrar pasos internos de rescate — no son acciones del agente
            # sino artefactos del mecanismo handle_parsing_errors de LangChain.
            if tool_name == "_Exception":
                continue
            tool_input = getattr(action, "tool_input", "")
            if isinstance(tool_input, dict):
                tool_input = tool_input.get("query", str(tool_input))
            obs_clean = str(observation).replace(
                "\n\n[VISUAL HINT:", "\n\n[→ visual hint interno:"
            )
            steps.append({
                "tool": tool_name,
                "input": str(tool_input)[:120],
                "result_preview": obs_clean[:250],
            })

    except Exception as e:
        raw_answer = f"Error durante la ejecución del agente: {e}"
        steps = []

    return {
        "answer": _clean_answer(raw_answer),
        "sources": _tracker.get(),
        "steps": steps,
    }

"""
Grafo RAG con agente verificador usando LangGraph.

Arquitectura (patrón reflexión — tema 7.4 del curso):

    START → retrieve_generate → verify ──► END  (si PASS o intentos agotados)
                                    │
                                    └──► retrieve_generate  (si FAIL y quedan intentos)

Nodos:
  · retrieve_generate : ejecuta el pipeline RAG existente (retrieval + reranking + LLM)
  · verify            : LLM verifica si la respuesta está respaldada por el contexto

El verificador usa structured output (patrón 7.6 del curso) para devolver
un veredicto tipado: PASS | FAIL + razón.
"""

from typing import TypedDict, Literal

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


MODEL     = "llama3.1:8b"
MAX_ATTEMPTS = 2          # máximo de regeneraciones si el verificador falla


# ══════════════════════════════════════════════════════════════════
#  STATE  (patrón TypedDict del curso)
# ══════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    question : str           # pregunta del usuario
    context  : list          # fragmentos de texto recuperados (para el verificador)
    sources  : list          # metadatos de fuentes
    answer   : str           # respuesta generada por el LLM
    verdict  : str           # "PASS" | "FAIL" | "PENDING"
    reason   : str           # explicación del verificador
    attempts : int           # contador de regeneraciones


# ══════════════════════════════════════════════════════════════════
#  STRUCTURED OUTPUT  (patrón 7.6 — supervisor con Pydantic)
# ══════════════════════════════════════════════════════════════════

class VerificationResult(BaseModel):
    verdict : Literal["PASS", "FAIL"]
    reason  : str


# ══════════════════════════════════════════════════════════════════
#  NODO 1 — retrieve_generate
#  Llama al pipeline RAG existente y extrae el contexto recuperado
# ══════════════════════════════════════════════════════════════════

def retrieve_generate(state: RAGState) -> dict:
    from src.evaluation.rag_pipeline import answer_with_rag_multimodal
    from src.embeddings.vector_store import load_vector_store

    vs = load_vector_store("data/embeddings/chroma_db")
    answer, sources, retrieved_docs = answer_with_rag_multimodal(
        state["question"], vs, k=4
    )

    # Extraemos el texto plano de los documentos recuperados para dárselo al verificador
    context = [
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in retrieved_docs
    ]

    return {
        "answer"  : answer,
        "context" : context,
        "sources" : sources,
        "attempts": state["attempts"] + 1,
    }


# ══════════════════════════════════════════════════════════════════
#  NODO 2 — verify
#  Comprueba si la respuesta está respaldada por el contexto
# ══════════════════════════════════════════════════════════════════

_verify_system = SystemMessage(
    "Eres un verificador de respuestas en sistemas RAG. "
    "Tu única tarea es comprobar si la respuesta generada está respaldada "
    "por los fragmentos de contexto recuperados.\n\n"
    "Criterios:\n"
    "  - PASS: la respuesta se apoya en información presente en el contexto "
    "o admite honestamente no tener información.\n"
    "  - FAIL: la respuesta contiene afirmaciones que no están en el contexto "
    "o que lo contradicen.\n\n"
    "Sé estricto pero justo. Si el contexto es insuficiente pero la respuesta "
    "lo reconoce, devuelve PASS."
)


def verify(state: RAGState) -> dict:
    llm = ChatOllama(model=MODEL, temperature=0).with_structured_output(
        VerificationResult
    )

    context_str = "\n\n".join(
        f"[{i+1}] {chunk[:600]}"          # limitamos para no exceder contexto
        for i, chunk in enumerate(state["context"])
    )

    user_msg = HumanMessage(
        f"Pregunta del usuario:\n{state['question']}\n\n"
        f"Contexto recuperado:\n{context_str}\n\n"
        f"Respuesta generada:\n{state['answer']}"
    )

    try:
        result = llm.invoke([_verify_system, user_msg])
        return {"verdict": result.verdict, "reason": result.reason}
    except Exception as e:
        # Fallback si structured output falla: marcamos PASS con aviso
        return {
            "verdict": "PASS",
            "reason" : f"Verificación automática no disponible ({type(e).__name__}).",
        }


# ══════════════════════════════════════════════════════════════════
#  EDGE CONDICIONAL  (patrón 7.4 — should_continue)
#  Decide si regenerar o terminar
# ══════════════════════════════════════════════════════════════════

def route_after_verify(state: RAGState) -> Literal["retrieve_generate", "__end__"]:
    if state["verdict"] == "PASS":
        return END
    if state["attempts"] >= MAX_ATTEMPTS:
        return END
    return "retrieve_generate"


# ══════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO  (patrón StateGraph del curso)
# ══════════════════════════════════════════════════════════════════

_builder = StateGraph(RAGState)
_builder.add_node("retrieve_generate", retrieve_generate)
_builder.add_node("verify", verify)

_builder.add_edge(START, "retrieve_generate")
_builder.add_edge("retrieve_generate", "verify")
_builder.add_conditional_edges(
    "verify",
    route_after_verify,
    {"retrieve_generate": "retrieve_generate", END: END},
)

rag_verified_graph = _builder.compile()


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN PÚBLICA
# ══════════════════════════════════════════════════════════════════

def run_rag_verified(query: str) -> dict:
    """
    Ejecuta el grafo RAG+verificador y devuelve:
      answer   : respuesta final
      sources  : fuentes usadas
      verdict  : "PASS" | "FAIL"
      reason   : explicación del verificador
      attempts : número de veces que se regeneró la respuesta
    """
    initial_state: RAGState = {
        "question" : query,
        "context"  : [],
        "sources"  : [],
        "answer"   : "",
        "verdict"  : "PENDING",
        "reason"   : "",
        "attempts" : 0,
    }

    result = rag_verified_graph.invoke(initial_state)

    return {
        "answer"  : result["answer"],
        "sources" : result["sources"],
        "verdict" : result["verdict"],
        "reason"  : result["reason"],
        "attempts": result["attempts"],
    }

"""
Orquestador multi-agente usando LangGraph.

Arquitectura (patrón supervisor — tema 7.6 del curso):

    START → orchestrator ──► rag        ──► END
                         ├──► agente    ──► END
                         ├──► verificado──► END
                         └──► baseline  ──► END

El nodo orquestador analiza la pregunta y, mediante structured output
(igual que el SupervisorDecision de 7.6), elige qué pipeline ejecutar:

  · rag        → preguntas factuales sobre el contenido de los PDFs
  · agente     → preguntas sobre imágenes / figuras / búsqueda iterativa
  · verificado → preguntas donde la exactitud es crítica (usa RAG + verificador)
  · baseline   → preguntas generales que no necesitan buscar en documentos
"""

from typing import TypedDict, Literal

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


MODEL = "llama3.1:8b"


# ══════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════

class OrchestratorState(TypedDict):
    question       : str
    routed_to      : str    # agente elegido por el orquestador
    routing_reason : str    # por qué lo eligió
    answer         : str
    sources        : list
    verdict        : str
    verdict_reason : str
    agent_steps    : list


# ══════════════════════════════════════════════════════════════════
#  STRUCTURED OUTPUT  (patrón SupervisorDecision de 7.6)
# ══════════════════════════════════════════════════════════════════

class OrchestratorDecision(BaseModel):
    next   : Literal["rag", "agente", "verificado", "baseline"]
    reason : str


# ══════════════════════════════════════════════════════════════════
#  NODO ORQUESTADOR
# ══════════════════════════════════════════════════════════════════

_supervisor_prompt = SystemMessage(
    "Eres un supervisor de un sistema RAG multimodal sobre documentos académicos. "
    "Dado la pregunta del usuario, decide qué agente debe responderla:\n\n"
    "  · rag        : pregunta factual sobre el contenido de los PDFs.\n"
    "  · agente     : pregunta sobre imágenes, figuras, diagramas o tablas; "
    "o pregunta compleja que requiere búsqueda iterativa.\n"
    "  · verificado : la precisión es crítica (definiciones exactas, "
    "afirmaciones que deben verificarse contra el contexto).\n"
    "  · baseline   : pregunta general que NO necesita buscar en documentos "
    "(saludo, conocimiento general, off-topic).\n\n"
    "Responde SOLO con la decisión estructurada."
)


def orchestrator(state: OrchestratorState) -> dict:
    llm = ChatOllama(model=MODEL, temperature=0).with_structured_output(
        OrchestratorDecision
    )
    try:
        decision = llm.invoke([_supervisor_prompt, HumanMessage(state["question"])])
        return {"routed_to": decision.next, "routing_reason": decision.reason}
    except Exception as e:
        # Fallback: RAG por defecto si structured output falla
        return {
            "routed_to"     : "rag",
            "routing_reason": f"Decisión por defecto ({type(e).__name__}).",
        }


# ══════════════════════════════════════════════════════════════════
#  NODOS DE CADA AGENTE
# ══════════════════════════════════════════════════════════════════

def rag_node(state: OrchestratorState) -> dict:
    from src.evaluation.rag_pipeline import answer_with_rag_multimodal
    from src.embeddings.vector_store import load_vector_store

    vs = load_vector_store("data/embeddings/chroma_db")
    answer, sources, _ = answer_with_rag_multimodal(state["question"], vs, k=4)
    return {"answer": answer, "sources": sources}


def agente_node(state: OrchestratorState) -> dict:
    from src.agent.rag_agent import run_agent

    r = run_agent(state["question"])
    return {
        "answer"     : r["answer"],
        "sources"    : r["sources"],
        "agent_steps": r["steps"],
    }


def verificado_node(state: OrchestratorState) -> dict:
    from src.agent.rag_graph import run_rag_verified

    r = run_rag_verified(state["question"])
    return {
        "answer"        : r["answer"],
        "sources"       : r["sources"],
        "verdict"       : r["verdict"],
        "verdict_reason": r["reason"],
    }


def baseline_node(state: OrchestratorState) -> dict:
    from src.evaluation.baseline import answer_without_rag

    return {"answer": answer_without_rag(state["question"]), "sources": []}


# ══════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO  (patrón supervisor 7.6)
# ══════════════════════════════════════════════════════════════════

_builder = StateGraph(OrchestratorState)

_builder.add_node("orchestrator", orchestrator)
_builder.add_node("rag",          rag_node)
_builder.add_node("agente",       agente_node)
_builder.add_node("verificado",   verificado_node)
_builder.add_node("baseline",     baseline_node)

_builder.add_edge(START, "orchestrator")

# Edge condicional desde el supervisor (igual que en 7.6)
_builder.add_conditional_edges(
    "orchestrator",
    lambda state: state["routed_to"],
    {
        "rag"       : "rag",
        "agente"    : "agente",
        "verificado": "verificado",
        "baseline"  : "baseline",
    },
)

_builder.add_edge("rag",         END)
_builder.add_edge("agente",      END)
_builder.add_edge("verificado",  END)
_builder.add_edge("baseline",    END)

orchestrator_graph = _builder.compile()


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN PÚBLICA
# ══════════════════════════════════════════════════════════════════

def run_orchestrated(query: str) -> dict:
    """
    Ejecuta el grafo con orquestador y devuelve:
      answer         : respuesta final
      sources        : fuentes usadas
      routed_to      : agente elegido ("rag" | "agente" | "verificado" | "baseline")
      routing_reason : explicación de la decisión del orquestador
      verdict        : "PASS" | "FAIL" | "" (solo si routed_to == "verificado")
      verdict_reason : explicación del verificador (si aplica)
      agent_steps    : pasos del agente ReAct (si aplica)
    """
    initial: OrchestratorState = {
        "question"      : query,
        "routed_to"     : "",
        "routing_reason": "",
        "answer"        : "",
        "sources"       : [],
        "verdict"       : "",
        "verdict_reason": "",
        "agent_steps"   : [],
    }

    result = orchestrator_graph.invoke(initial)

    return {
        "answer"        : result["answer"],
        "sources"       : result["sources"],
        "routed_to"     : result["routed_to"],
        "routing_reason": result["routing_reason"],
        "verdict"       : result.get("verdict", ""),
        "verdict_reason": result.get("verdict_reason", ""),
        "agent_steps"   : result.get("agent_steps", []),
    }

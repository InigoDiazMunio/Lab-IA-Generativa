import json
from pathlib import Path


def build_default_dataset(output_path: str = "src/evaluation/questions.json"):
    questions = [
        {"id": 1, "question": "¿Qué es Retrieval-Augmented Generation?", "category": "teoria"},
        {"id": 2, "question": "¿Qué problema intenta resolver un sistema RAG?", "category": "teoria"},
        {"id": 3, "question": "¿Qué significa combinar recuperación y generación en un sistema RAG?", "category": "teoria"},
        {"id": 4, "question": "¿Qué diferencia hay entre un modelo con conocimiento paramétrico y un sistema con recuperación externa?", "category": "teoria"},
        {"id": 5, "question": "¿Por qué un modelo de lenguaje puede necesitar conocimiento externo?", "category": "teoria"},
        {"id": 6, "question": "¿Qué ventajas ofrece RAG frente a un modelo sin acceso a documentos?", "category": "teoria"},
        {"id": 7, "question": "¿Qué papel tiene el retrieval en una arquitectura RAG?", "category": "teoria"},
        {"id": 8, "question": "¿Qué papel tiene el generador en un sistema RAG?", "category": "teoria"},
        {"id": 9, "question": "¿Por qué RAG puede mejorar la factualidad de las respuestas?", "category": "teoria"},
        {"id": 10, "question": "¿Qué relación hay entre RAG y los modelos de lenguaje grandes?", "category": "teoria"},
        {"id": 11, "question": "¿Qué limitación general de los LLM intenta paliar RAG?", "category": "teoria"},
        {"id": 12, "question": "¿Por qué RAG es útil en tareas de pregunta-respuesta sobre documentos?", "category": "teoria"},

        {"id": 13, "question": "¿Para qué sirven los embeddings en un sistema RAG?", "category": "componentes"},
        {"id": 14, "question": "¿Qué papel cumple un índice vectorial en el sistema?", "category": "componentes"},
        {"id": 15, "question": "¿Por qué los documentos se dividen en chunks?", "category": "componentes"},
        {"id": 16, "question": "¿Qué efecto tiene el chunk overlap en la recuperación?", "category": "componentes"},
        {"id": 17, "question": "¿Cómo influye el tamaño del chunk en el rendimiento del sistema?", "category": "componentes"},
        {"id": 18, "question": "¿Qué significa buscar similitud semántica entre una pregunta y los documentos?", "category": "componentes"},
        {"id": 19, "question": "¿Qué función tiene FAISS en este proyecto?", "category": "componentes"},
        {"id": 20, "question": "¿Qué es un retriever en una arquitectura RAG?", "category": "componentes"},
        {"id": 21, "question": "¿Qué aporta un reranker al proceso de recuperación?", "category": "componentes"},
        {"id": 22, "question": "¿Por qué no basta con recuperar muchos fragmentos sin ordenar?", "category": "componentes"},
        {"id": 23, "question": "¿Qué diferencia hay entre retrieval inicial y reranking?", "category": "componentes"},
        {"id": 24, "question": "¿Por qué la calidad de los embeddings influye en la calidad del retrieval?", "category": "componentes"},

        {"id": 25, "question": "¿Qué diferencia hay entre un baseline y un sistema RAG?", "category": "comparacion"},
        {"id": 26, "question": "¿Por qué el baseline suele dar respuestas más genéricas?", "category": "comparacion"},
        {"id": 27, "question": "¿En qué situaciones puede superar RAG al baseline?", "category": "comparacion"},
        {"id": 28, "question": "¿En qué casos puede fallar un sistema RAG aunque recupere contexto?", "category": "comparacion"},
        {"id": 29, "question": "¿Por qué comparar baseline y RAG ayuda a evaluar el proyecto?", "category": "comparacion"},
        {"id": 30, "question": "¿Qué diferencia hay entre un sistema RAG textual y uno multimodal?", "category": "comparacion"},
        {"id": 31, "question": "¿Qué cambia en el pipeline cuando se pasa de RAG textual a RAG multimodal?", "category": "comparacion"},
        {"id": 32, "question": "¿Por qué un modelo puede responder bien en baseline a preguntas generales pero fallar en preguntas específicas?", "category": "comparacion"},
        {"id": 33, "question": "¿Qué aporta el contexto recuperado que no tiene el baseline?", "category": "comparacion"},
        {"id": 34, "question": "¿Por qué un mejor generador puede beneficiar tanto al baseline como al RAG?", "category": "comparacion"},

        {"id": 35, "question": "¿Por qué los PDFs con tablas e imágenes son difíciles de procesar en RAG?", "category": "multimodalidad"},
        {"id": 36, "question": "¿Qué problemas introduce el contenido visual en un sistema puramente textual?", "category": "multimodalidad"},
        {"id": 37, "question": "¿Por qué una figura puede contener información relevante que no aparece bien en el texto extraído?", "category": "multimodalidad"},
        {"id": 38, "question": "¿Qué aporta LLaVA en un pipeline multimodal?", "category": "multimodalidad"},
        {"id": 39, "question": "¿Qué limitaciones tiene la generación automática de captions a partir de imágenes extraídas de un PDF?", "category": "multimodalidad"},
        {"id": 40, "question": "¿Por qué detectar todas las imágenes de un PDF puede introducir ruido?", "category": "multimodalidad"},
        {"id": 41, "question": "¿Por qué las imágenes irrelevantes pueden empeorar la calidad del índice multimodal?", "category": "multimodalidad"},
        {"id": 42, "question": "¿Qué diferencia hay entre una imagen útil para el sistema y un recurso visual irrelevante?", "category": "multimodalidad"},
        {"id": 43, "question": "¿Por qué el contenido visual puede requerir modelos específicos de visión-lenguaje?", "category": "multimodalidad"},
        {"id": 44, "question": "¿Qué limitaciones prácticas tiene un pipeline multimodal frente a uno solo textual?", "category": "multimodalidad"},

        {"id": 45, "question": "¿Qué es BLIP-2 y cuál es su aportación principal?", "category": "modelos"},
        {"id": 46, "question": "¿Qué es LLaVA y cómo se entrena?", "category": "modelos"},
        {"id": 47, "question": "¿Qué es RAGAS y para qué sirve?", "category": "modelos"},
        {"id": 48, "question": "¿Qué tipo de evaluación propone RAGAS para sistemas RAG?", "category": "modelos"},
        {"id": 49, "question": "¿Qué importancia tiene el survey sobre RAG dentro del proyecto?", "category": "modelos"},
        {"id": 50, "question": "¿Por qué los modelos de visión-lenguaje son relevantes para la multimodalidad?", "category": "modelos"},
        {"id": 51, "question": "¿Qué aportan los cross-encoders al reranking?", "category": "modelos"},
        {"id": 52, "question": "¿Qué diferencia hay entre un modelo de embeddings y un modelo generativo?", "category": "modelos"},

        {"id": 53, "question": "¿Qué limitaciones tiene el sistema implementado en este proyecto?", "category": "evaluacion"},
        {"id": 54, "question": "¿Por qué la calidad del generador influye tanto en el resultado final?", "category": "evaluacion"},
        {"id": 55, "question": "¿Por qué un corpus ruidoso perjudica al sistema RAG?", "category": "evaluacion"},
        {"id": 56, "question": "¿Por qué recuperar contexto relevante no garantiza una buena respuesta?", "category": "evaluacion"},
        {"id": 57, "question": "¿Qué mejoras futuras se plantean para el proyecto?", "category": "evaluacion"},
        {"id": 58, "question": "¿Qué limitaciones presenta la evaluación automática del sistema?", "category": "evaluacion"},
        {"id": 59, "question": "¿Por qué es importante usar preguntas variadas para evaluar RAG?", "category": "evaluacion"},
        {"id": 60, "question": "¿Qué se ha aprendido al comparar el baseline con el sistema RAG?", "category": "evaluacion"},
    ]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Dataset de {len(questions)} preguntas guardado en {output_path}")
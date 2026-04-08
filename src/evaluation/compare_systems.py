import json


def compare_results(path="experiments/evaluation_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for item in results:
        print("=" * 80)
        print(f"Pregunta: {item['question']}")
        print("\n--- Respuesta RAG ---")
        print(item["rag_answer"])
        print("\n--- Fuentes RAG ---")
        for src in item["rag_sources"]:
            print(f"- {src['source_file']} | página {src['page']}")

        print("\n--- Respuesta baseline ---")
        print(item["baseline_answer"])
        print("=" * 80)


if __name__ == "__main__":
    compare_results()
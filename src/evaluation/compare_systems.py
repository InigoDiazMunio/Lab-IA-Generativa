import json


def compare_results(path="experiments/evaluation_results.json"):
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for item in results:
        print("=" * 100)
        print(f"ID: {item['id']}")
        print(f"Pregunta: {item['question']}")
        print(f"Categoría: {item.get('category', '')}")

        print("\n--- RESPUESTA RAG ---")
        print(item["rag_answer"])

        print("\n--- FUENTES RAG ---")
        for src in item["rag_sources"]:
            print(f"- Archivo: {src['source_file']} | Página: {src['page']}")
            print(f"  Preview: {src['content_preview'][:150]}...")

        print("\n--- RESPUESTA BASELINE ---")
        print(item["baseline_answer"])
        print("=" * 100 + "\n")


if __name__ == "__main__":
    compare_results()
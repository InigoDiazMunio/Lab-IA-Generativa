from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness


def run_ragas_eval(questions, answers, contexts, references=None, llm=None):
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
        ],
        llm=llm,
    )

    return result
from transformers import pipeline


class LocalLLM:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=256
        )

    def generate(self, prompt: str) -> str:
        result = self.pipe(prompt)
        return result[0]["generated_text"]


_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance


def generate_answer(prompt: str):
    llm = get_llm()
    return llm.generate(prompt)
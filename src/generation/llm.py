from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LocalLLM:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return " ".join(text.split()).strip()


_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance


def generate_answer(prompt: str):
    llm = get_llm()
    return llm.generate(prompt)
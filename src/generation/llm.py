import json
import urllib.request
import urllib.error


class LocalLLM:
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://127.0.0.1:11434/api/generate",
        timeout: int = 180,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                text = result.get("response", "").strip()
                return " ".join(text.split())

        except urllib.error.URLError as e:
            raise RuntimeError(
                "No se pudo conectar con Ollama. "
                "Comprueba que Ollama está instalado, abierto y que el modelo "
                f"'{self.model_name}' está descargado."
            ) from e


_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance


def generate_answer(prompt: str):
    llm = get_llm()
    return llm.generate(prompt)
"""
Captioning de imágenes vía Ollama (LLaVA).
"""

import base64
import json
import urllib.request
import urllib.error
from pathlib import Path


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LLAVA_MODEL = "llava"


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_llava(image_path: str, prompt: str, timeout: int = 120) -> str:
    image_b64 = _image_to_base64(image_path)

    payload = {
        "model": LLAVA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result.get("response", "").strip()


def generate_basic_caption(image_path: str) -> str:
    """
    Genera una descripción en castellano de la imagen usando LLaVA.
    Filtra iconos y logos pequeños por tamaño de archivo.
    """
    image_path = str(image_path)

    # Filtrar imágenes muy pequeñas (iconos, logos)
    try:
        size_bytes = Path(image_path).stat().st_size
        if size_bytes < 10000:
            return ""
    except Exception:
        pass

    prompt = (
        "IMPORTANTE: Responde ÚNICAMENTE en español. No uses portugués ni inglés.\n\n"
        "Describe en español lo que ves en esta imagen extraída de un artículo científico. "
        "Si es un diagrama, explica su estructura. "
        "Si es una tabla, resume los datos. "
        "Si es una fotografía de ejemplo, describe qué aparece. "
        "\n\nRecuerda: tu respuesta debe estar en español, 2-3 frases máximo."
    )

    try:
        caption = _call_llava(image_path, prompt)

        if not caption:
            return ""

        return caption

    except urllib.error.URLError:
        print(f"  [AVISO] Ollama no disponible para {Path(image_path).name}")
        return ""
    except Exception as e:
        print(f"  [AVISO] Error captioning {Path(image_path).name}: {e}")
        return ""


def generate_captions_for_page(extracted_images: list) -> list:
    results = []
    for item in extracted_images:
        caption = generate_basic_caption(item["image_path"])
        if caption:
            results.append({**item, "caption": caption})
    return results
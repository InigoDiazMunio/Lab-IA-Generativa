from pathlib import Path


def generate_basic_caption(image_path: str) -> str:
    """
    Placeholder simple.
    Más adelante  integrar un modelo real de image captioning.
    """
    image_name = Path(image_path).name
    return f"Imagen extraída del documento: {image_name}"
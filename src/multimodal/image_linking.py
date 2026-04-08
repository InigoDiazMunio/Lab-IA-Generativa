def link_images_to_pages(extracted_images):
    """
    Devuelve una estructura que asocia imágenes con páginas del PDF.
    """
    linked = []

    for item in extracted_images:
        linked.append({
            "pdf": item["pdf"],
            "page": item["page"],
            "image_path": item["image_path"]
        })

    return linked
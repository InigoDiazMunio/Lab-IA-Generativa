from pathlib import Path
import fitz


def extract_images_from_pdf(pdf_path: str, output_folder: str = "data/processed/images"):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_folder) / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    saved_images = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            image_name = f"page_{page_index + 1}_img_{img_index + 1}.{ext}"
            image_path = output_dir / image_name

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            saved_images.append({
                "pdf": pdf_path.name,
                "page": page_index + 1,
                "image_path": str(image_path)
            })

    return saved_images
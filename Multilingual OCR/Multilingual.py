import os
import json
import torch
import tempfile
import warnings
import logging
import numpy as np
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from transformers import AutoImageProcessor
from transformers.models.detr import DetrForSegmentation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import gradio as gr

# ---------- Setup ----------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Load DETR Layout Model ----------
logging.info("Loading DETR model...")
img_proc = AutoImageProcessor.from_pretrained("cmarkea/detr-layout-detection")
model = DetrForSegmentation.from_pretrained("cmarkea/detr-layout-detection")
model.eval()
logging.info("DETR model loaded.")

# ---------- Load Surya OCR ----------
logging.info("Loading Surya OCR...")
surya_detector = DetectionPredictor()
surya_recognizer = RecognitionPredictor()
logging.info("Surya OCR loaded.")

# ---------- Main Function ----------
def detect_and_ocr(pdf_file):
    pages = convert_from_path(pdf_file.name, dpi=150)
    threshold = 0.4
    annotated_images = []
    full_json = []

    for page_idx, img in enumerate(pages):
        logging.info(f"Processing page {page_idx + 1}...")
        orig_img = img.convert("RGB")
        inputs = img_proc(orig_img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        result = img_proc.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[orig_img.size[::-1]]
        )[0]

        draw = ImageDraw.Draw(orig_img)
        page_info = {"page": page_idx + 1, "blocks": []}

        for i, (box, label, score) in enumerate(zip(result["boxes"], result["labels"], result["scores"])):
            box = [int(x) for x in box.tolist()]
            label_name = model.config.id2label[label.item()]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 10), f"{label_name} ({score:.2f})", fill="red")

            # Crop box for OCR
            crop = img.crop(box)

            try:
                ocr_result = surya_recognizer([crop], det_predictor=surya_detector)
                text_lines = ocr_result[0].text_lines
                text = "\n".join([line.text.strip() for line in text_lines])
            except Exception as e:
                logging.warning(f"OCR failed on page {page_idx + 1}, box {i + 1}: {e}")
                text = ""

            page_info["blocks"].append({
                "label": label_name,
                "score": float(score),
                "box": box,
                "ocr_text": text
            })

        full_json.append(page_info)
        annotated_images.append(orig_img)

    # ---------- Save Annotated PDF ----------
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, "annotated_output.pdf")
    json_path = os.path.join(temp_dir, "annotated_output.json")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    for img in annotated_images:
        img_width, img_height = img.size
        aspect = img_height / img_width
        target_width, target_height = letter
        if aspect > 1:
            new_height = target_height
            new_width = target_height / aspect
        else:
            new_width = target_width
            new_height = target_width * aspect

        x = (target_width - new_width) / 2
        y = (target_height - new_height) / 2
        c.drawImage(ImageReader(img), x, y, width=new_width, height=new_height)
        c.showPage()
    c.save()

    # ---------- Save JSON ----------
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(full_json, jf, indent=2, ensure_ascii=False)

    return pdf_path, json_path

# ---------- Gradio Interface ----------
iface = gr.Interface(
    fn=detect_and_ocr,
    inputs=gr.File(label="Upload PDF", file_types=[".pdf"]),
    outputs=[
        gr.File(label="Download Annotated PDF"),
        gr.File(label="Download OCR JSON")
    ],
    title="Layout Detection + Surya OCR",
    description="Detect layout elements (tables, text, etc.) and run OCR inside each using Surya OCR."
)

if __name__ == "__main__":
    logging.info("Launching Gradio app...")
    iface.launch()

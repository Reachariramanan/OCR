from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import gradio as gr

# ✅ Load model and processor
model_id = "microsoft/Florence-2-base"
print("Loading Florence 2 model...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 🧠 Inference function
def run_ocr(image, task_type):
    if task_type not in ["<OCR>", "<OCR_WITH_REGION>"]:
        return "Invalid task selected."

    prompt = task_type

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate on CPU
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

    # Decode and post-process
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    if task_type == "<OCR>":
        return parsed_answer["<OCR>"].strip()
    else:
        return str(parsed_answer)

# 🎛️ Gradio UI
gr.Interface(
    fn=run_ocr,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(["<OCR>", "<OCR_WITH_REGION>"], label="Select OCR Mode")
    ],
    outputs="text",
    title="🧠 Florence OCR Playground",
    description="Upload an image and run OCR or OCR with regions using Florence-2-base from Microsoft."
).launch(share=True)

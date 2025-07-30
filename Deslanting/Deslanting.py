import gradio as gr
import cv2
import numpy as np
from deslant_img import deslant_img
import os

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cut_images = []
    fixed_width = gray.shape[1]
    fixed_height = 148

    num_crops = gray.shape[0] // fixed_height
    remaining_height = gray.shape[0] % fixed_height

    for i in range(num_crops):
        y_start = i * fixed_height
        y_end = y_start + fixed_height
        cropped_img = gray[y_start:y_end, :]
        cut_images.append(cropped_img)

    if remaining_height > 0:
        cropped_img = gray[-remaining_height:, :]
        cut_images.append(cropped_img)

    deslanted_images = []
    for line_img in cut_images:
        res = deslant_img(line_img)
        deslanted_img = cv2.resize(res.img, (fixed_width, line_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        deslanted_images.append(deslanted_img)

    final_height = sum(img.shape[0] for img in deslanted_images)
    final_image = np.zeros((final_height, fixed_width), dtype=np.uint8)

    y_offset = 0
    for deslanted_img in deslanted_images:
        h, w = deslanted_img.shape
        final_image[y_offset:y_offset + h, :w] = deslanted_img
        y_offset += h

    return final_image

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Slanted Image"),
    outputs=gr.Image(type="numpy", label="Deslanted Image"),
    title="Handwriting Deslanting Tool",
    description="Upload a slanted handwritten image. The app will segment and deslant each section, returning the corrected version."
)

if __name__ == "__main__":
    iface.launch()

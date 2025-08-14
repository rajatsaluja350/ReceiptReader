
import streamlit as st
import requests
from PIL import Image
from transformers import pipeline
import io

# Title
st.title("AI Receipt Reader")

# Upload image
uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

# OCR.Space API key (replace with your own key)
OCR_API_KEY = "K86414890888957"  # rajat's API key

def extract_text_from_image(image_bytes):
    url = "https://api.ocr.space/parse/image"
    payload = {
        "isOverlayRequired": False,
        "apikey": OCR_API_KEY,
        "language": "eng",
    }
    files = {
        "file": ("receipt.jpg", image_bytes),
    }
    response = requests.post(url, data=payload, files=files)
    result = response.json()
    if result.get("IsErroredOnProcessing"):
        return None
    return result["ParsedResults"][0]["ParsedText"]

def parse_receipt_with_llm(text):
    prompt = f"Extract the following fields from this receipt text:\n\n{text}\n\nFields: Vendor, Date, Total, Items (with quantity and price)."
    llm = pipeline("text2text-generation", model="google/flan-t5-base")
    result = llm(prompt, max_new_tokens=256)[0]["generated_text"]
    return result

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Receipt", use_column_width=True)

    with st.spinner("Extracting text with OCR..."):
        image_bytes = uploaded_file.read()
        extracted_text = extract_text_from_image(image_bytes)

    if extracted_text:
        st.subheader(" Extracted Text")
        st.text(extracted_text)

        with st.spinner("Parsing with LLM..."):
            parsed_output = parse_receipt_with_llm(extracted_text)

        st.subheader(" Parsed Receipt Data")
        st.text(parsed_output)
    else:
        st.error("Failed to extract text from the image.")

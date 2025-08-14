import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import pipeline

# Load the LLM pipeline
llm = pipeline("text2text-generation", model="google/flan-t5-base")

# OCR.Space API configuration
OCR_API_KEY = "K86414890888957"
OCR_API_URL = "https://api.ocr.space/parse/image"

# Function to extract text from image using OCR.Space
def extract_text_from_image(image_bytes):
    response = requests.post(
        OCR_API_URL,
        files={"file": image_bytes},
        data={"apikey": OCR_API_KEY, "language": "eng", "isOverlayRequired": False},
        verify=False  # Disable SSL verification for compatibility
    )
    result = response.json()
    if result.get("IsErroredOnProcessing"):
        return "Error extracting text from image."
    parsed_results = result.get("ParsedResults")
    if parsed_results:
        return parsed_results[0].get("ParsedText", "")
    return "No text found."

# Function to parse receipt text using LLM
def parse_receipt_text(text):
    prompt = (
        "Extract the following fields from the receipt text:
"
        "Vendor:
Date:
Total:
Items:

"
        f"Receipt Text:
{text}

"
        "Structured Output:"
    )
    result = llm(prompt, max_new_tokens=200)
    return result[0]['generated_text']

# Streamlit UI
st.title("🧾 AI-Powered Receipt Reader")

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Receipt", use_column_width=True)

    with st.spinner("Extracting text using OCR..."):
        image_bytes = uploaded_file.getvalue()
        extracted_text = extract_text_from_image(image_bytes)

    st.subheader("📝 Extracted Text")
    st.text(extracted_text)

    with st.spinner("Parsing receipt using LLM..."):
        structured_output = parse_receipt_text(extracted_text)

    st.subheader("📊 Parsed Receipt Data")
    st.text(structured_output)

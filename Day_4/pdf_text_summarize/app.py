from transformers import pipeline
import gradio as gr
import PyPDF2

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def text_summarize(pdf_file):
    # Read PDF
    text = ""
    reader = PyPDF2.PdfReader(pdf_file.name)
    for page in reader.pages:
        text += page.extract_text()

    # summarization logic
    article = text[:1500]  #(just demo: first 1500 chars)
    response = summarizer(article)
    return response[0]['summary_text']

# creating the user interface using gradio
demo = gr.Interface(
    fn=text_summarize,
    inputs=gr.File(type="filepath", file_types=[".pdf"]),
    outputs="text",
    title="Text Summarizer",
    description="Upload a PDF to summarize"

)

demo.launch()

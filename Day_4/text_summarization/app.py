from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

import gradio as gr

def text_summarize(article):
    response = summarizer(article)
    return response[0]['summary_text']

demo = gr.Interface(fn=text_summarize, inputs="text", outputs="text", title='text Summarize', description='Enter Article to Summarize')
demo.launch()


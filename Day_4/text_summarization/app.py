from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

import gradio as gr

#from gradio_pdf import PDF

# with gr.Blocks() as demo:
#     pdf = PDF(label="Upload a PDF", interactive=True)
#     name = gr.Textbox()
#     pdf.upload(lambda f: f, pdf, name)
#
# demo.launch()

def text_summarize(article):
    response = summarizer(article)
    return response[0]['summary_text']

demo = gr.Interface(fn=text_summarize(), inputs="text", outputs="text", title='text Summarize', description='Enter Article to Summarize')
demo.launch()

#print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)[0]['summary_text'])


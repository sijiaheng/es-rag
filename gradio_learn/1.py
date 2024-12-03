import gradio as gr

def greet(name,B):
    return "hello"+name+"!"

iface = gr.Interface(fn = greet,inputs=[gr.Textbox(lines=5,placeholder="name here",label="name:"),gr.Radio(["1","2"])],outputs="text")

iface.launch(share=True)
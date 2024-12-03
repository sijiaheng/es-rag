import gradio as gr

input_list = [
    gr.Audio(sources=["microphone","upload"],type="numpy"),#音频输入，可以录制或上传音频文件，可使用sources来限制输入方式
    gr.Checkbox()
]
import gradio as gr

def combine_sentence(sen1, sen2):
    return sen1 + ' ' + sen2

# Define the Interface
demo = gr.Interface(
    fn = combine_sentence,
    inputs = [
        gr.Textbox(label = "Input 1"),
        gr.Textbox(label = 'Input 2')
    ],
    outputs = gr.Textbox(value = "", label = "Output")
)

# launch the interface
demo.launch(server_name='0.0.0.0', server_port = 7860)
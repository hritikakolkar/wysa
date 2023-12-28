import gradio as gr
from inference import Inference

def predict(tweet):
    return inference.get_batch_inference([tweet])[0]

inference = Inference(model_path="weights/v001", model_name= "pytorch_model.bin")

iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Emotion Classifier for Tweets",
    description="Enter a Tweet to classify the emotion towards a brand or product."
)

if __name__ == "__main__":
    iface.launch()

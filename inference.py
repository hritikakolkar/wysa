import os
import torch
from transformers import RobertaTokenizer
from src.models.Classifier import EmotionClassifier
from typing import List


class Inference():
    def __init__(self, model_path, model_name, pretrained_model_path = "weights/twitter-roberta-base-sentiment-latest", max_length= 70):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = EmotionClassifier(num_classes_emotion=3, pretrained_model_path= pretrained_model_path)
        self.model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
        self.model.to(self.device)
        self.max_length = max_length
        self.id_to_emotion = ['No emotion toward brand or product', 'Positive emotion', 'Negative emotion']

    def get_batch_inference(self, batch_tweets: List)-> List:
        batch = self.tokenizer(batch_tweets, padding=True, truncation=True, max_length = 70, return_tensors="pt")
        batch.to(self.device)
        with torch.no_grad():
            output = self.model(**batch)
        return list(map(lambda x: self.id_to_emotion[x], torch.argmax(output, dim=1).tolist()))
    
if __name__ == '__main__':
    inference = Inference(model_path="weights/v001", model_name= "pytorch_model.bin")
    
    print(inference.get_batch_inference(["I love IPad"]))
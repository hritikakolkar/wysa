import torch.nn as nn
from transformers import RobertaModel, RobertaConfig


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes_emotion, pretrained_model_path):
        super(EmotionClassifier, self).__init__()
        self.model_config = RobertaConfig.from_pretrained(pretrained_model_path)
        self.pretrained_model = RobertaModel.from_pretrained(pretrained_model_path, config= self.model_config)
        self.dropout = nn.Dropout(0.3)
        # We can use single classifier with classes = num_classes_entity + num_classes_emotion, but using this approach for simplification
        self.classifier_emotion = nn.Linear(self.pretrained_model.config.hidden_size, num_classes_emotion)
    
    def forward(self, input_ids, attention_mask):
        pooler_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        pooler_output= nn.ReLU()(pooler_output)
        pooler_output= self.dropout(pooler_output)

        # Output for emotion classification
        emotion_output = self.classifier_emotion(pooler_output)
        # No need of output_probs as using nn.CrossEntropyLoss
        # emotion_output_probs = nn.Softmax(dim=1)(emotion_output)

        return emotion_output
    

class EntityEmotionClassifier(nn.Module):
    def __init__(self, num_classes_entity, num_classes_emotion, pretrained_model_path):
        super(EntityEmotionClassifier, self).__init__()
        self.model_config = RobertaConfig.from_pretrained(pretrained_model_path)
        self.pretrained_model = RobertaModel.from_pretrained(pretrained_model_path, config= self.model_config)
        self.dropout = nn.Dropout(0.3)
        # We can use single classifier with classes = num_classes_entity + num_classes_emotion, but using this approach for simplification
        self.classifier_entity = nn.Linear(self.pretrained_model.config.hidden_size, num_classes_entity)
        self.classifier_emotion = nn.Linear(self.pretrained_model.config.hidden_size, num_classes_emotion)
    
    def forward(self, input_ids, attention_mask):
        pooler_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        pooler_output= nn.ReLU()(pooler_output)
        pooler_output= self.dropout(pooler_output)

        # Output for entity classification
        output_entity = self.classifier_entity(pooler_output)
        # No need of output_probs as using nn.CrossEntropyLoss
        # output_entity_probs = nn.Softmax(dim=1)(output_entity)

        # Output for emotion classification
        output_emotion = self.classifier_emotion(pooler_output)
        # No need of output_probs as using nn.CrossEntropyLoss
        # output_emotion_probs = nn.Softmax(dim=1)(output_emotion)

        return output_entity, output_emotion
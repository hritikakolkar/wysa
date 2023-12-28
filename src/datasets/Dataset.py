import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, tweet, emotion, tokenizer, max_length):
        self.tweet = tweet
        self.emotion = emotion
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.tweet[idx],
            max_length = self.max_length,
            add_special_tokens=True,
            padding = "max_length",
            return_attention_mask=True,
            return_tensors='pt',
            )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emotion_labels': torch.tensor(self.emotion[idx], dtype=torch.long)
        }

        return item


class EntityEmotionDataset(Dataset):
    def __init__(self, tweet, entity, emotion, tokenizer):
        self.tweet = tweet
        self.entity = entity
        self.emotion = emotion
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.tweet[idx],
            add_special_tokens=True,
            padding = "longest",
            return_attention_mask=True,
            return_tensors='pt',
            )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'entity_labels': torch.tensor(self.entity[idx], dtype=torch.long),
            'emotion_labels': torch.tensor(self.emotion[idx], dtype=torch.long)
        }

        return item

import os
import pprint
import yaml
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import preprocessing, compute_metrics
from src.utils import parse_yaml
from src.datasets.Dataset import EmotionDataset
from src.models.Classifier import EmotionClassifier
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer


def train_and_eval(model, train_dataloader, test_dataloader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    model.to(config["device"])
    wandb.watch(model, criterion, log="all", log_freq=10)
    samples_count = 0
    global_step = 0
    best_f1_score = 0.0
    for epoch in tqdm(range(config["num_epochs"])):
        model.train()
        log_batch_loss = 0
        log_batch_emotion_output = np.array([])
        log_batch_emotion_labels = np.array([])
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(config["device"])
            attention_mask = batch['attention_mask'].to(config["device"])
            emotion_labels = batch['emotion_labels'].to(config["device"])

            
            optimizer.zero_grad()
            emotion_output = model(input_ids, attention_mask)

            loss = criterion(emotion_output, emotion_labels)

            loss.backward()
            optimizer.step()
            
            global_step += 1
            samples_count += len(emotion_output)

            log_batch_loss += float(loss)
            log_batch_emotion_output = np.hstack((log_batch_emotion_output, torch.argmax(emotion_output, dim=1).cpu().numpy()))
            log_batch_emotion_labels = np.hstack((log_batch_emotion_labels, emotion_labels.cpu().numpy()))
            if (global_step + 1) % config["log_batch_step"] == 0:
                precision, recall, accuracy, f1 = compute_metrics(log_batch_emotion_output, log_batch_emotion_labels, config)
                wandb.log({
                    "epoch": epoch,
                    "loss" : log_batch_loss/config["log_batch_step"],
                    "global_step" : global_step,
                    "samples_count" : samples_count, 
                    "train_precision": precision,
                    "train_recall": recall,
                    "train_f1_score": f1,
                    "train_accuracy": accuracy
                }, step = global_step)
                log_batch_loss = 0
                log_batch_emotion_output = np.array([])
                log_batch_emotion_labels = np.array([])

        model.eval()
        with torch.no_grad():
            loss = 0
            test_metrics = np.array([0.0, 0.0, 0.0, 0.0])
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(config["device"])
                attention_mask = batch['attention_mask'].to(config["device"])
                emotion_labels = batch['emotion_labels'].to(config["device"])

                emotion_output = model(input_ids, attention_mask)

                batch_metrics = np.array(compute_metrics(torch.argmax(emotion_output, dim=1), emotion_labels, config))
                test_metrics += batch_metrics
                loss += criterion(emotion_output, emotion_labels)
            test_precision, test_recall, test_accuracy, test_f1 = test_metrics/ len(test_dataloader)
            wandb.log({
                "epoch": epoch,
                "test_loss" : float(loss)/len(test_dataloader),
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1_score": test_f1,
                "test_accuracy": test_accuracy
            }, step = global_step)

        if epoch == 0 or (epoch+1)%config["model_save_epoch"] == 0 or epoch == config["num_epochs"]-1:
            torch.save(model.state_dict(), os.path.join(config["model_save_path"],f"iter_epoch_{str(epoch).zfill(3)}.bin"))
        
        if test_f1 >= best_f1_score:
            best_epoch = epoch
            best_f1_score = test_f1
            torch.save(model.state_dict(), os.path.join(config["model_save_path"],"pytorch_model.bin"))
    print(f"Best Epoch for best f1_score of {best_f1_score} is {best_epoch}. Model saved as pytorch_model.bin in the respective version")

def make_config(config):
    train = preprocessing(pd.read_excel("data/dataset.xlsx", sheet_name = "Train"), config= config, train=True)
    # test = preprocessing(pd.read_excel("data/dataset.xlsx", sheet_name = "Test"), train=False)
    x_train, x_test, y_train, y_test = train_test_split(train["tweet"].to_numpy(), train["emotion"].to_numpy(), test_size=config["test_size"], random_state= config["seed"])

    tokenizer = RobertaTokenizer.from_pretrained(config["pretrained_model_path"])
    tokenizer.save_pretrained(config["model_save_path"])
    tokenizer.save_pretrained(config["model_save_path"])
    
    train_dataset = EmotionDataset(tweet= x_train, emotion= y_train, tokenizer= tokenizer, max_length= config["max_length"])
    test_dataset = EmotionDataset(tweet= x_test, emotion= y_test, tokenizer= tokenizer, max_length= config["max_length"])

    train_dataloader = DataLoader(dataset= train_dataset, batch_size= 8)
    test_dataloader = DataLoader(dataset= test_dataset, batch_size= 8)

    model = EmotionClassifier(config["num_classes_emotion"], config["pretrained_model_path"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    return model, train_dataloader, test_dataloader, criterion, optimizer

def model_pipeline(hyperparameters, name):
    # tell wandb to get started
    with wandb.init(project="wysa", name= name, config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        # make the model, data, and optimization problem
        model, train_dataloader, test_dataloader, criterion, optimizer = make_config(config)
        
        # and use them to train the model
        train_and_eval(model, train_dataloader, test_dataloader, criterion, optimizer, config)
    
if __name__=='__main__':
    config = parse_yaml()
    # run the training pipeline
    print(f"Version of Model : {config['version']}")
    print(f"Config of Model : ")
    pprint.pprint(config["config"])
    wandb.login() # wandb authentication
    model_pipeline(config["config"], config["version"])

    # saving config file
    with open(os.path.join(config["config"]["model_save_path"], "training_config.yaml") ,'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
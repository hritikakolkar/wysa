import sys
import yaml
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def parse_yaml():
    # Check if the config file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file_path>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Replace {{ version }} with the actual version value
    config['config']['model_save_path'] = config['config']['model_save_path'].replace('{{ version }}', config['version'])
    config['config']['learning_rate'] = float(config['config']['learning_rate'])
    # Print the modified config
    return config

def preprocessing(df, config = None, train=True):
    if train:
        df = df.rename(columns={
            'tweet_text': 'tweet',
            'emotion_in_tweet_is_directed_at': 'entity', # entity means brand, product or service
            'is_there_an_emotion_directed_at_a_brand_or_product': 'emotion'
        })
        df = df[df["tweet"].notna()].reset_index(drop = True)
        df["tweet"] = df["tweet"].str.replace(pat = r'[^a-zA-Z0-9#@\!$%^&*(){}:\-\'\":;,\.?/\s]', repl = ' ', regex=True)
        df["tweet"] = df["tweet"].str.replace(pat = r'[:;,\.?/\s]{2,}', repl = ' ', regex= True)
        df["tweet"] = df["tweet"].str.strip()
        df = df.drop("entity", axis = 1)
        df["emotion"] = df["emotion"].replace(config["emotion_to_id"])
    else:
        df = df.rename(columns={
            'Tweet': 'tweet'
        })
        df = df[df["tweet"].notna()].reset_index(drop = True)
        df["tweet"] = df["tweet"].str.replace(pat = r'[^a-zA-Z0-9#@\!$%^&*(){}:\-\'\":;,\.?/\s]', repl = ' ', regex=True)
        df["tweet"] = df["tweet"].str.replace(pat = r'[:;,\.?/\s]{2,}', repl = ' ', regex= True)
        df["tweet"] = df["tweet"].str.strip()
    return df

def compute_metrics(predictions, targets, config):
    # Convert tensors to numpy arrays
    if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
        predictions_np = predictions
        targets_np = targets
    else:
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

    labels = np.unique(predictions_np)
    # Compute metrics
    precision = precision_score(targets_np, predictions_np, average= config["average"], labels= labels, zero_division= config["zero_division"])
    recall = recall_score(targets_np, predictions_np, average= config["average"], labels= labels, zero_division= config["zero_division"])
    accuracy = accuracy_score(targets_np, predictions_np)
    f1 = f1_score(targets_np, predictions_np, average= config["average"], labels= labels, zero_division= config["zero_division"])

    return precision, recall, accuracy, f1
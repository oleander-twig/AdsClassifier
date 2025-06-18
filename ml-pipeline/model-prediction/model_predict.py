from config import create_config, Config
import boto3
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import numpy as np
import string
import pymorphy2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os

from cls import init_my_model, My_Model, My_dataset, init_my_optimizer, init_my_criterion 



def model_predict() -> None:
    cfg = create_config()

    s3 = boto3.client(
    cfg.s3.service_name,
    aws_access_key_id=cfg.s3.aws_access_key_id,
    aws_secret_access_key= cfg.s3.aws_secret_access_key,
    endpoint_url=cfg.s3.public_url
    )
    s3.download_file(cfg.s3.bucket_name, 'output.csv', 'output.csv')
    s3.download_file(cfg.s3.bucket_name, 'categories.csv', 'categories.csv')

    data = pd.read_csv('output.csv')
    categories = pd.read_csv('categories.csv')

    vectorizer, (df_train, df_val) = prepare_data(data,categories)

    # fix seed
    set_seed()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device - ", device)

    model = init_my_model(device, 1040, len(categories['name']))

    train_data = My_dataset(df_train, "count")
    val_data = My_dataset(df_val, "count")

    batch_size = 2048
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = init_my_optimizer(model)
    criterion = init_my_criterion(df_train, device)
    
    train_my_model(epochs=20, model=model, train_loader=train_loader, val_loader=val_loader, device=device, criterion=criterion, optimizer=optimizer)

    dump_trained_artifacts(s3, cfg.s3.bucket_name, model, vectorizer)

    return 



def prepare_data(data, categories):
    # Step 1: drop punctuation
    # punctiation, currencies, etc.
    symbols_to_avoid = string.punctuation + '—«»“”‘’$€£¥₽' 
    data['input_non_punct'] = data["input"].apply(lambda x: drop_punctuation(x, symbols_to_avoid))

    # Step 2: normalize text
    morph = pymorphy2.MorphAnalyzer()
    data['tokens'] = data['input_non_punct'].apply(lambda x: normalize_text(x, morph))

    # Step 3: split train/test
    df_train, df_val = train_test_split(data,
                                        test_size=0.2,
                                        random_state=42)

    # Step 4: vectorize
    vectorizer = CountVectorizer(analyzer='word', min_df=10)
    df_train['count_vectorizer'] = [vec for vec in vectorizer.fit_transform(df_train['tokens']).toarray()]
    df_val['count_vectorizer'] = [vec for vec in vectorizer.transform(df_val['tokens']).toarray()]

    # Step 5: encode categories
    # str:int
    category_encoded_map = dict()
    for i, c in enumerate(categories['name']):
        category_encoded_map[c] = i

    # int:str
    category_encoded_map_reversed = dict()
    for category, category_id in category_encoded_map.items():
        category_encoded_map_reversed[category_id] = category

    
    # helper function to encode category
    def encode_category(category_id):
        vec_encoded = np.zeros(len(categories))
        vec_encoded[category_id] = 1
        return vec_encoded

    df_train['category_encoded'] = df_train['category'].apply(lambda x: encode_category(category_encoded_map[x]))
    df_val['category_encoded'] = df_val['category'].apply(lambda x: encode_category(category_encoded_map[x]))

    return vectorizer, (df_train, df_val)



def drop_punctuation(line, sym):
  for p in sym:
    if p in line:
        line = line.replace(p, '')
  return line


def normalize_text(text, morph):
  return ' '.join([morph.parse(word)[0].normal_form for word in text.split() if word != ''])


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def train_my_model(epochs, model, train_loader, val_loader, device, criterion, optimizer):
    for epoch_num in range(epochs):

        # for accuracy
        train_correct_predicted_count = 0
        val_correct_predicted_count = 0

        # for recall
        train_correct_predicted_map = dict()
        train_target_map = dict()
        val_correct_predicted_map = dict()
        val_target_map = dict()

        # for precision
        train_all_predicted_map = dict()
        val_all_predicted_map = dict()

        train_loss = 0
        val_loss = 0

        model.train()
        for data, hubs in train_loader:
            data = data.to(device)
            hubs = hubs.to(device)
            output = model(data)
            loss = criterion(output, hubs)
            train_loss += loss.item()
            output = output.detach().cpu().numpy()
            hubs = hubs.detach().cpu().numpy()

            # итерируемся по текстам из батча
            for i in range(len(output)):
                output_i = output[i]
                category_id_predicted = np.argmax(output_i)
                category_id_target = np.argmax(hubs[i])
                if category_id_predicted == category_id_target:
                    train_correct_predicted_count += 1
                    if category_id_predicted in train_correct_predicted_map.keys():
                        train_correct_predicted_map[category_id_predicted] += 1
                    else:
                        train_correct_predicted_map[category_id_predicted] = 1

                if category_id_predicted in train_all_predicted_map.keys():
                    train_all_predicted_map[category_id_predicted] += 1
                else:
                    train_all_predicted_map[category_id_predicted] = 1
                if category_id_target in train_target_map.keys():
                    train_target_map[category_id_target] += 1
                else:
                    train_target_map[category_id_target] = 1

            model.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for data, hubs in val_loader:
            data = data.to(device)
            hubs = hubs.to(device)
            output = model(data)
            loss = criterion(output, hubs)
            val_loss += loss.item()
            output = output.detach().cpu().numpy()
            hubs = hubs.detach().cpu().numpy()
            for i in range(len(output)):
                output_i = output[i]
                category_id_predicted = np.argmax(output_i)
                category_id_target = np.argmax(hubs[i])

                if category_id_predicted == category_id_target:
                    val_correct_predicted_count += 1

                    if category_id_predicted in val_correct_predicted_map.keys():
                        val_correct_predicted_map[category_id_predicted] += 1
                    else:
                        val_correct_predicted_map[category_id_predicted] = 1

                if category_id_predicted in val_all_predicted_map.keys():
                    val_all_predicted_map[category_id_predicted] += 1
                else:
                    val_all_predicted_map[category_id_predicted] = 1

                if category_id_target in val_target_map.keys():
                    val_target_map[category_id_target] += 1
                else:
                    val_target_map[category_id_target] = 1

        train_recall = recall_count(train_correct_predicted_map, train_target_map)
        val_recall = recall_count(val_correct_predicted_map, val_target_map)

        train_pr = precision_count(train_correct_predicted_map, train_all_predicted_map)
        val_pr = precision_count(val_correct_predicted_map, val_all_predicted_map)


        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {train_loss} Train Precision: {train_pr} Train Recall: {train_recall} \
            " + "\n" + " "*len(str(epoch_num + 1)) +
            f"         | Vsl Loss {val_loss} Val Precision: {val_pr} | Val Recall: {val_recall}")
        

def recall_count(predicted_map, target_map):
  res = []
  for category_id in predicted_map.keys():
    res.append(predicted_map[category_id] / target_map[category_id])
  return np.mean(res) # макро усреднение


def precision_count(predicted_correct_map, predicted_all_map):
  res = []
  for category_id in predicted_correct_map.keys():
    res.append(predicted_correct_map[category_id] / predicted_all_map[category_id])
  return np.mean(res) # макро усреднение


def dump_trained_artifacts(s3, bucket_name, model, vectorizer):
    torch.save(model, "./model.pth")
    with open('./vectorizer.bin', 'wb') as f:
        pickle.dump(vectorizer, file=f)

    s3.upload_file('./model.pth', bucket_name, 'model.pth')
    s3.upload_file('./vectorizer.bin', bucket_name, 'vectorizer.bin')


if __name__ == "__main__":
    model_predict()

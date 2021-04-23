from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import json
import pathlib
from importlib import import_module

def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            )

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)

    return np.array(output_pred).flatten()

def inference_roberta_softmax(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device)
            )

        logits = outputs[0].detach().cpu().numpy()
        # print(outputs)
        # print(outputs[0].shape)
        # logits = torch.argmax(outputs[0],1)
        # logits = logits.detach().cpu().numpy()
        # result = logits
        # result = np.argmax(logits, axis=1)

        output_pred.extend(logits)

    return np.array(output_pred)

def load_test_dataset(dataset_dir, tokenizer, data_set:bool):
    if data_set:
        test_dataset = load_data(dataset_dir)
    else:
        test_dataset = pd.read_csv(dataset_dir, delimiter='\t')
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def model_soft(location,test_file,data_set:bool):
    location_pre = pathlib.Path(location).parents[0]
    with open(os.path.join(location_pre, 'config.json')) as f:
        json_data = json.load(f)

    model_dict = {
        "bert_base_multilingual_cased": "bert-base-multilingual-cased",
        "xlm_roberta_large": "xlm-roberta-large",
        "xlm_roberta_base": "xlm-roberta-base",
        "koelectra_base_v3_discriminator": "monologg/koelectra-base-v3-discriminator",
        "koelectra_small_v3_discriminator": "monologg/koelectra-small-v3-discriminator",
        "kobert": "monologg/kobert"

    }

    """
      주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    TOK_NAME = model_dict[json_data['model']]
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens":["α","@","β","#"]})

    # load model and tokenizer

    print(json_data)
    MODEL_NAME = model_dict[json_data['model']]
    MODEL_TYPE = ''
    if 'bert-base' in MODEL_NAME:
        MODEL_TYPE = 'Bert'
    elif 'xlm-roberta' in MODEL_NAME:
        # MODEL_TYPE = 'Roberta'
        MODEL_TYPE = 'XLMRoberta'
    elif 'koelectra' in MODEL_NAME:
        MODEL_TYPE = 'Electra'
    elif 'kobert' in MODEL_NAME:
        MODEL_TYPE = 'Bert'
    else:
        RuntimeError()
    # setting model hyperparameter
    model_module = getattr(import_module("transformers"), MODEL_TYPE + "ForSequenceClassification")
    model = model_module.from_pretrained(location)
    model.parameters
    model = model.to(device)

    # load test datset
    # test_dataset_dir = "../input/data/test/test.tsv"
    test_dataset_dir = test_file
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, data_set)
    test_dataset = RE_Dataset(test_dataset ,test_label)

    # predict answer
    # pred_answer = inference(model, test_dataset, device)
    output = inference_roberta_softmax(model, test_dataset, device)
    return output


def main(args):
    pred_soft1 = model_soft("./results_e10/final","../input/data/test/test.tsv",data_set=True)
    pred_soft2 = model_soft("./results_04/checkpoint-900","../input/data/test/test.tsv",data_set=True)
    pred_soft3 = model_soft("./results_ab/final","../input/data/test/test+ab_p.tsv",data_set=False)
    pred_soft4 = model_soft("./results_03/checkpoint-1000","../input/data/test/test.tsv",data_set=True)

    pred_soft = pred_soft1 + pred_soft2*1.1 + pred_soft3 + pred_soft4
    # print(pred_soft)
    pred_soft = pred_soft
    pred = np.argmax(pred_soft, axis=1)
    # print(pred)

    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(pred, columns=['pred'])
    output.to_csv('./prediction/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', type=str, default="./results/final")
    args = parser.parse_args()
    print(args)
    main(args)
  

import argparse
import json
import os
import random
from importlib import import_module

import pickle as pickle
import pandas as pd
import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from transformers import ElectraConfig, ElectraModel
from transformers import RobertaConfig, RobertaModel
# from transformers.trainer_pt_utils import LabelSmoother


import wandb

from load_data import *
from loss import Cross_FocalLoss, Smooth_FocalLoss

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = Cross_FocalLoss()
        loss = loss_fct(logits,labels)

        return (loss, outputs) if return_outputs else loss

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    wandb.log({'val/acc': acc})
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(args):
    seed_everything(args.seed)

    # wandb logger
    wandb.init(project="Pstage2_automl1", config=vars(args))

    # load model and tokenizer
    model_dict = {
        "bert_base_multilingual_cased":"bert-base-multilingual-cased",
        "xlm_roberta_large":"xlm-roberta-large",
        "xlm_roberta_base":"xlm-roberta-base",
        "koelectra_base_v3_discriminator": "monologg/koelectra-base-v3-discriminator",
        "koelectra_small_v3_discriminator": "monologg/koelectra-small-v3-discriminator",
        "kobert": "monologg/kobert"

    }

    MODEL_NAME = model_dict[args.model]
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens":["α","@","β","#"]})

    # -- logging
    if not os.path.exists('./results'):
        os.mkdir('./results')
    with open(os.path.join('./results','config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # load dataset
    # train_dataset = load_data(args.data_dir)
    train_dataset = pd.read_csv(args.data_dir, delimiter='\t')
    N = len(train_dataset)
    val_ratio = 0.1
    N_train = N - int((N * val_ratio))
    train_dataset, dev_dataset = train_dataset[:N_train], train_dataset[N_train:]
    # train_dataset, dev_dataset = train_dataset[N-N_train:], train_dataset[:N-N_train]
    # dev_dataset = load_data("./dataset/train/dev.tsv")
    train_label = train_dataset['label'].values
    dev_label = dev_dataset['label'].values
    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting model hyperparameter
    config_module = getattr(import_module("transformers"), MODEL_TYPE + "Config")
    model_config = config_module.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42
    model_module = getattr(import_module("transformers"), MODEL_TYPE + "ForSequenceClassification")
    model = model_module.from_pretrained(MODEL_NAME, config=model_config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        # fp16=True,
        # dataloader_num_workers=4,
        report_to='wandb',
        output_dir='./results',          # output directory
        save_total_limit=2,              # number of total save model.
        save_steps=900,                 # model saving step.
        num_train_epochs=args.epochs,              # total number of training epochs
        learning_rate=args.lr,               # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_steps=300,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps = 100,            # evaluation step.
        dataloader_num_workers=4,
        run_name='automl1',  # name of the W&B run
    )

    trainer = MultilabelTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics  # define metrics function
    )

    # trainer = Trainer(
    #     model=model,                         # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=RE_train_dataset,         # training dataset
    #     eval_dataset=RE_dev_dataset,             # evaluation dataset
    #     compute_metrics=compute_metrics         # define metrics function
    # )

    # train model
    trainer.train()
    trainer.save_model('./results/final/')


def main(args):
    train(args)

if __name__ == '__main__':
    wandb.login()

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=float, default=6)
    parser.add_argument('--batch_size', type=int, default=16, help='per_device_train_batch_size')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased')

    # environ directories
    parser.add_argument('--data_dir', type=str, default="../input/data/train/train.tsv")

    args = parser.parse_args()

    main(args)

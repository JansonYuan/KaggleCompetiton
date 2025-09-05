import os
os.environ['HF_ENDPOINT'] = 'hf-mirror.com'

import os
import sys
import pdb
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, log_loss
import openpyxl
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.special import softmax
from transformers import set_seed
from models import CustomModel, CustomModelRank
from utils_datasets import UnifiedSFTDataset, UnifiedSFTDatasetRank, SelfDataCollatorWithPadding
set_seed(42)
answer_map = {0:'model_a', 1:'model_b', 2: 'tie'}


token = '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'
model_name = './deberta-v3-base'
save_dir = './output/deberta_v3_base/'
dtype = None
load_in_4bit = False
if_use_rank = False
MAX_LEN= 1024
#model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
if if_use_rank:
    model = CustomModelRank(model_name)
else:
    model = CustomModel(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = SelfDataCollatorWithPadding(tokenizer)
def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)

df = pd.read_csv('train.csv')
df['prompt'] = df['prompt'].apply(lambda x:process(x))
df['response_a'] = df['response_a'].apply(lambda x:process(x))
df['response_b'] = df['response_b'].apply(lambda x:process(x))
df['label'] = np.argmax(df[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)
train, test = train_test_split(df, random_state=64)
train_instructs, train_response_a, train_response_b, train_labels = train['prompt'].tolist(), \
                                                                    train['response_a'].tolist(), \
                                                                    train['response_b'].tolist(), \
                                                                    train['label'].tolist()
test_instructs, test_response_a, test_response_b, test_labels = test['prompt'].tolist(), \
                                                                    test['response_a'].tolist(), \
                                                                    test['response_b'].tolist(), \
                                                                    test['label'].tolist()


if if_use_rank:
    train_tokenized_datasets = UnifiedSFTDatasetRank(train_instructs, train_response_a, train_response_b, train_labels, tokenizer, MAX_LEN)
    test_tokenized_datasets = UnifiedSFTDatasetRank(test_instructs, test_response_a, test_response_b, test_labels, tokenizer, MAX_LEN)
else:
    train_tokenized_datasets = UnifiedSFTDataset(train_instructs, train_response_a, train_response_b, train_labels, tokenizer, MAX_LEN)
    test_tokenized_datasets = UnifiedSFTDataset(test_instructs, test_response_a, test_response_b, test_labels, tokenizer, MAX_LEN)

print(train_tokenized_datasets[0])
print(tokenizer.decode(train_tokenized_datasets[0]['input_ids_0']))
print(tokenizer.decode(train_tokenized_datasets[0]['input_ids_1']))
training_args = TrainingArguments(
    output_dir=save_dir, 
    learning_rate=5e-5,
    num_train_epochs = 5,
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    eval_steps  = 2000,
    save_steps = 2000,
    warmup_steps = 0,
    gradient_accumulation_steps=1,
    save_total_limit = 3,
    load_best_model_at_end = True,
    fp16=True,
    metric_for_best_model='log_loss',
    greater_is_better=False,
    #gradient_checkpointing=True
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    prob = softmax(logits, axis=-1)
    p = precision_score(labels, predictions, average='micro')
    r = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    log_loss_score = log_loss(labels, prob)
    return {'log_loss': log_loss_score, 'precision_score': p, 'recall_score': r, 'f1_score': f1}

def compute_metrics_rank(eval_pred):
    logits, labels = eval_pred
    # labels [2, 1, 0]
    # logits [1.5, 0.7, -0.03]
    prob = np.where(prob<0, 0, prob)
    preds = np.zeros(prob.shape[0], 3)
    preds[prob<1,0]
    # pred 1.5 -> 0, sigmoid(0.5), 1-sigmoid(0.5)
    # pred 0.7 -> 0, 1-sigmoid(0.5), sigmoid(0.5)
    log_loss_score = log_loss(labels, preds)
    return {'log_loss': log_loss_score, }


class MYTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        if if_use_rank:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss




trainer = MYTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

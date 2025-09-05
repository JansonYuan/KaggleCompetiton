import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
os.environ['HF_HUB_CACHE'] = '/data/guanhan/_huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/guanhan/_huggingface'
os.environ['HF_ENDPOINT'] = 'https://fuxi-demos-huggingface-mirror.us-sjc38-eng-general.k8s.tesla.com'
os.environ['REQUESTS_CA_BUNDLE'] = 'Tesla_Root.crt'
os.environ['CURL_CA_BUNDLE'] = 'Tesla_Root.crt'
import math
from torch.cuda import amp
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoConfig, TrainingArguments, BitsAndBytesConfig, Trainer
from transformers import MistralForSequenceClassification, DataCollatorWithPadding, set_seed
from datasets import Dataset
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from timm.utils import ModelEmaV2
from adversarial_training import AWP, FGM
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from utils import create_scheduler, create_custom_deberta_optimizer
from metrics import compute_metrics
from llm_models import LLMAESModel, AESMistral
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import gc
from accelerate import init_empty_weights

a = 0  # 2.948
b = 1.092
data_path = 'external_data_with_compare_v5.csv'
exp_name = 'exp15'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True
TRAINING_MODEL_PATH = 'h2oai/h2o-danube-1.8b-base'  # Undi95/Meta-Llama-3-8B-hf 'meta-llama/Llama-2-7b-hf' 'mistralai/Mistral-7B-v0.1'  'h2oai/h2o-danube-1.8b-base' #"microsoft/deberta-v3-base"  # your model path
token = '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'
use_kbit = False
use_lora = True
task_type = 'reg'
use_kappa_loss = False
batch_size = 4
eval_steps = 600
epochs = 1
n_folds = 7
train_folds = [2, 3, 4, 5, 6]
lr = 10e-5
head_lr = 5e-5
if_awp = False
if_fgm = False
is_llm = True
max_grad_norm = 1000
gradient_accumulation_steps = 1
discriminative_learning_rate = False
discriminative_learning_rate_num_groups = 12
discriminative_learning_rate_decay_rate = 0.99
ema_decay = 0.99
TRAINING_MAX_LENGTH = 2000  # I use 1280 locally
save_name = TRAINING_MODEL_PATH.replace('/', '-')

data = pd.read_csv(data_path)
data['full_text'] = data['full_text_x']
data['score'] = data['score_x']
#data = data.sample(n=100, random_state=64).reset_index(drop=True)
data["fold"] = -1
X = data["full_text"]
y = data["score"]
skf = StratifiedGroupKFold(n_splits=n_folds)
for i, (train_index, val_index) in enumerate(skf.split(X, y, data['prompt_name'])):
    data.loc[val_index, "fold"] = i
data['score'] = data['score'] - a

tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH, token=token)
tokenizer.pad_token = tokenizer.eos_token


@dataclass
class AESTrainingArguments(TrainingArguments):
    adverserial_training: bool = field(default=False,
                                       metadata={"help": "Wheter to use adverserial_training or not to use."})
    adverserial_method: str = field(default='AWP', metadata={"help": "Specify the adverserial_method to use."})
    adverserial_learning_rate: float = field(default=1e-3,
                                             metadata={"help": "Learning rate to use for adverserial training."})
    adverserial_epsilon: float = field(default=1e-6, metadata={"help": "Epsilon rate to use for adverserial training."})
    adverserial_training_start_epoch: int = field(default=1, metadata={"help": "Epoch to start adverserial training."})
    discriminative_learning_rate: bool = field(default=False, metadata={
        "help": "Wheter to use discriminative_learning_rate or not to use."})
    discriminative_learning_rate_num_groups: int = field(default=1, metadata={
        "help": "Number of groups for which we should use the same learning rate."})
    discriminative_learning_rate_decay_rate: float = field(default=0.9, metadata={
        "help": "Exponential decay rate per layer to apply for discriminative learning rate."})
    head_lr: float = field(default=1e-4, metadata={
        "help": "Learning rate to use for task specific head during args.discriminative_learning_rate==True."})
    adam_optim_bits: int = field(default=None, metadata={
        "help": "Number of bits to use during optimization. Use 32 for standard Adam and 8 for 8-bit Adam. If None use Standard AdamW"})


def tokenize(example, tokenizer):
    tokenized = tokenizer(example['full_text'],
                          return_offsets_mapping=True,
                          truncation=True, max_length=TRAINING_MAX_LENGTH)

    return {
        **tokenized,
        "labels": torch.tensor(example['score'], dtype=torch.float32),
    }


def cls_tokenize(example, tokenizer):
    tokenized = tokenizer(example['full_text'],
                          truncation=True, max_length=TRAINING_MAX_LENGTH)

    return {
        **tokenized,
        "labels": torch.tensor(example['score'], dtype=torch.long),
    }


def get_model(model_name):
    config = AutoConfig.from_pretrained(model_name, token=token)
    config.num_labels = 1
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    config.attention_dropout = 0
    config.problem_type = "regression"
    config.use_cache = True
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    with init_empty_weights():
        bert_model = AESMistral.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            # quantization_config=bnb_config,
            token=token
        )

    bert_model.config.pad_token_id = bert_model.config.eos_token_id

    peft_config = LoraConfig(
        lora_alpha=16,  # regularization
        lora_dropout=0.,
        r=32,  # attention heads
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    bert_model.enable_input_require_grads()
    bert_model = get_peft_model(bert_model, peft_config)
    for n, param in bert_model.named_parameters():
        if 'bilstm' in n or 'last_fc' in n:
            param.requires_grad = True
    return bert_model


def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids

    y_pred = predictions + 1.0
    y_pred = [i.round().astype(int) for i in y_pred]

    y_true = labels + 1.0
    y_true = [i.round().astype(int) for i in y_true]
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    results = {
        'cohen_kappa_score': score,
    }
    return results


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def preprocess(sample, text=False, infer_mode=False, max_seq=TRAINING_MAX_LENGTH, return_tensors=None):
    sys_prompt = "Please read the following essay and assign a score of 1,2,3,4,5,6 where 6 is the best. Output only a single number with no explanation.\n\n"
    prompt = sample["full_text"]
    if infer_mode:
        answer = ""
    else:
        answer = str(sample["score"])

    messages = [
        {"role": "user", "content": sys_prompt + prompt},
        {"role": "assistant", "content": f"\n\nThe score is: " + answer}
    ]
    formatted_sample = tokenizer.apply_chat_template(messages, tokenize=False)
    if infer_mode: formatted_sample = formatted_sample.replace("</s>", "")

    tokenized_sample = tokenizer(formatted_sample, padding=True, return_tensors=return_tensors,
                                 truncation=True, add_special_tokens=False, max_length=max_seq)

    if return_tensors == "pt":
        tokenized_sample["labels"] = tokenized_sample["input_ids"].clone()
    else:
        tokenized_sample["labels"] = tokenized_sample["input_ids"].copy()

    if text:
        return formatted_sample
    else:
        return tokenized_sample


def train_function_with_trainer(OUTPUT_DIR, args, fold):
    model = get_model(TRAINING_MODEL_PATH)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_ds = Dataset.from_pandas(data[data['fold'] != fold])
    print('total train data is *****')
    print(len(train_ds))
    if task_type == 'cls':
        train_ds = train_ds.map(cls_tokenize, fn_kwargs={"tokenizer": tokenizer},
                                num_proc=2).select_columns(['input_ids', 'attention_mask', 'labels'])
    elif task_type == 'instruct':
        train_ds = train_ds.map(preprocess, num_proc=4,
                                remove_columns=['essay_id', 'full_text', 'score'])
    else:
        train_ds = train_ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer},
                                num_proc=2).select_columns(['input_ids', 'attention_mask', 'labels'])
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator)
    val_ds = Dataset.from_pandas(data[data['fold'] == fold])
    print('total val data is *****')
    print(len(val_ds))
    if task_type == 'cls':
        val_ds = val_ds.map(cls_tokenize, fn_kwargs={"tokenizer": tokenizer, }, num_proc=2).select_columns(
            ['input_ids', 'attention_mask', 'labels'])
    elif task_type == 'instruct':
        val_ds = val_ds.map(preprocess, num_proc=4,
                            remove_columns=['essay_id', 'full_text', 'score'])
    else:
        val_ds = val_ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, }, num_proc=2).select_columns(
            ['input_ids', 'attention_mask', 'labels'])
    val_dataloader = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False, drop_last=False,
                                collate_fn=collator)
    set_seed(42)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    torch.save(model.bilstm.state_dict(), args.output_dir+'/bilstm.pt')
    torch.save(model.last_fc.state_dict(), args.output_dir + '/last_fc.pt')
    output = trainer.predict(val_ds)
    predictions, labels = output.predictions, output.label_ids
    predictions = (predictions + 1).clip(1, 6).round()

    part_oof = data[data['fold'] == fold].copy().reset_index(drop=True)
    part_oof['pred'] = predictions
    part_oof = part_oof[['score', 'pred']]
    if task_type == 'reg':
        part_oof['score'] = part_oof['score'] + 1
    return part_oof


def main():
    res = []

    data['score'] = data['score'] - 1
    for fold in train_folds:
        # if fold == 1:
        #     batch_size = 4
        #     eval_steps = 600
        # else:
        #     batch_size = 4
        #     eval_steps = 600
        OUTPUT_DIR = f'output_{save_name}_{exp_name}'  # your output path
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # val_ds = val_ds.class_encode_column("group")

        args = AESTrainingArguments(
            output_dir=f'{OUTPUT_DIR}/fold_{fold}',
            #fp16=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.0,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            logging_steps=100,
            save_steps=eval_steps,
            #optim="paged_adamw_32bit",
            save_total_limit=1,
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            lr_scheduler_type='cosine',
            metric_for_best_model="cohen_kappa_score",
            greater_is_better=True,
            discriminative_learning_rate=discriminative_learning_rate,
            discriminative_learning_rate_num_groups=discriminative_learning_rate_num_groups,
            discriminative_learning_rate_decay_rate=discriminative_learning_rate_decay_rate,
            head_lr=head_lr,
            # deepspeed="ds_config_zero2.json"
            gradient_checkpointing=True,
            weight_decay=0.,
            max_grad_norm=0.3,
        )

        part_oof = train_function_with_trainer(f'{OUTPUT_DIR}/fold_{fold}', args, fold)
        res.append(part_oof)
        torch.cuda.empty_cache()

    oof = pd.concat(res)
    if task_type == 'cls':
        oof['score'] = oof['score'] + 1
    else:
        oof['score'] = oof['score'] + a
    oof.to_csv(f'{OUTPUT_DIR}/oof.csv', index=None)

    y_true = oof['score']
    y_pred = oof['pred']
    total_score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f'total cv is {total_score}')


if __name__ == '__main__':
    main()

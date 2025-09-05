import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['HF_HUB_CACHE'] = '/data/guanhan/_huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/guanhan/_huggingface'
os.environ['HF_ENDPOINT'] = 'https://fuxi-demos-huggingface-mirror.us-sjc38-eng-general.k8s.tesla.com'
os.environ['REQUESTS_CA_BUNDLE'] = 'Tesla_Root.crt'
os.environ['CURL_CA_BUNDLE'] = 'Tesla_Root.crt'
from torch.cuda import amp
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorWithPadding, set_seed
from datasets import Dataset
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from timm.utils import ModelEmaV2
from adversarial_training import AWP, FGM
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from utils import create_scheduler, create_custom_deberta_optimizer
from metrics import compute_metrics
from models import AESModel
from tokenizers import AddedToken
a = 0 # 2.948
b = 1.092
use_scale = False
data_path = 'data_with_title.csv'
exp_name = 'exp14'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True
TRAINING_MODEL_PATH = 'microsoft/deberta-v3-small'  # 'h2oai/h2o-danube-1.8b-base' #"microsoft/deberta-v3-base"  # your model path
batch_size = 4
epoches = 5
n_folds = 7
lr = 1e-5
head_lr = 1e-4
if_awp = False
if_fgm = False
max_grad_norm = 1000
gradient_accumulation_steps = 1
discriminative_learning_rate = True
discriminative_learning_rate_num_groups = 6
discriminative_learning_rate_decay_rate = 0.99
ema_decay = 0.99
TRAINING_MAX_LENGTH = 2000  # I use 1280 locally
save_name = TRAINING_MODEL_PATH.replace('/', '-')

data = pd.read_csv(data_path)
data["fold"] = -1
X = data["full_text"]
y = data["score"]
skf = StratifiedGroupKFold(n_splits=n_folds)
for i, (train_index, val_index) in enumerate(skf.split(X, y, groups=data["prompt_name"])):
    data.loc[val_index, "fold"] = i
data['score'] = data['score'] - a

tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)
tokenizer.add_tokens([AddedToken("\n", normalized=False)])
tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])

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
    tokenized = tokenizer(example['full_text'], return_offsets_mapping=True,
                          truncation=True, max_length=TRAINING_MAX_LENGTH)

    return {
        **tokenized,
        "labels": torch.tensor(example['score'], dtype=torch.float32),
    }


def eval_one_epoch(model, val_dataloader):
    model.eval()
    val_labels = []
    val_preds = []
    for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        for key, value in batch.items():
            batch[key] = value.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        if isinstance(outputs, tuple):
            predictions = outputs[1]
        else:
            predictions = outputs.logits

        val_preds.append(predictions.cpu().numpy())
        val_labels.append(batch['labels'].cpu().numpy().reshape(-1,1))
    val_logits = np.vstack(val_preds, ).reshape(-1)
    val_gt = np.vstack(val_labels, ).reshape(-1)

    if use_scale:
        y_true = val_gt * 5 + 1
        y_pred = (val_logits * 5 + 1).clip(1, 6).round()
    else:
        y_true = val_gt + 1
        y_pred = (val_logits + 1).clip(1, 6).round()
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return score, y_pred


def train_function(OUTPUT_DIR, args, fold):

    model = AESModel(TRAINING_MODEL_PATH)
    model.bert_model.resize_token_embeddings(len(tokenizer))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_ds = Dataset.from_pandas(data[data['fold'] != fold])

    train_ds = train_ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer},
                            num_proc=8).select_columns(['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator)
    val_ds = Dataset.from_pandas(data[data['fold'] == fold])
    val_ds = val_ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, }, num_proc=8).select_columns(
        ['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                                collate_fn=collator)
    set_seed(42)

    model = model.to(device)
    model_ema = ModelEmaV2(model, decay=ema_decay, device=None)
    optimizer = create_custom_deberta_optimizer(args, model)
    num_examples = len(train_ds)
    num_update_steps_per_epoch = num_examples // (
            batch_size * 4)
    max_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        args.num_train_epochs * num_update_steps_per_epoch)
    scheduler = create_scheduler(args, model, max_steps, optimizer)
    best_score = 0
    best_pred = None
    if if_awp:
        print('Enable AWP')
        awp = AWP(model, optimizer, adv_lr=0.001, adv_eps=0.001)
    if if_fgm:
        print('Enable FGM')
        fgm = FGM(model)
    awp_start = 2
    scaler = amp.GradScaler(enabled=use_amp)
    for epoch in range(epoches):
        model.train()

        total_loss = 0

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            for key, value in batch.items():
                batch[key] = value.to(device)

            if if_awp and epoch >= awp_start:
                awp.perturb()
            with amp.autocast(use_amp):
                outputs = model(**batch)
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            scaler.scale(loss).backward()
            if if_awp:
                awp.restore()
            if if_fgm:
                fgm.attack()  # 在embedding上添加对抗扰动
                outputs = model(**batch)
                if isinstance(outputs, tuple):
                    loss_adv = outputs[0]
                else:
                    loss_adv = outputs.loss
                scaler.scale(loss_adv).backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                model_ema.update(model)
        score, y_pred = eval_one_epoch(model, val_dataloader)
        if score > best_score:
            torch.save(model.state_dict(), str(f"./{OUTPUT_DIR}/best_model_fold{fold}.pth"))
            best_score = score
            best_pred = y_pred
        print(f'fold_{fold} current score is ********')
        print(score)
        print(f'fold_{fold} best score is ********')
        print(best_score)
    part_oof = data[data['fold'] == fold].copy().reset_index(drop=True)
    part_oof['pred'] = best_pred
    part_oof = part_oof[['essay_id', 'score', 'pred']]
    return part_oof


def main():
    res = []
    if use_scale:
        data['score'] = (data['score'] - 1) / 5
    else:
        data['score'] = data['score'] - 1
    for fold in range(n_folds):
        OUTPUT_DIR = f'output_{save_name}_{exp_name}'  # your output path
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # val_ds = val_ds.class_encode_column("group")

        args = AESTrainingArguments(
            output_dir=OUTPUT_DIR,
            # fp16=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=100,
            learning_rate=lr,
            num_train_epochs=4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=4,
            report_to="none",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            lr_scheduler_type='cosine',
            metric_for_best_model="f1",
            greater_is_better=True,
            #optim="adafactor",
            discriminative_learning_rate=discriminative_learning_rate,
            discriminative_learning_rate_num_groups=discriminative_learning_rate_num_groups,
            discriminative_learning_rate_decay_rate=discriminative_learning_rate_decay_rate,
            head_lr=head_lr,
            # deepspeed="ds_config_zero2.json"
            # gradient_checkpointing=True,
            # weight_decay=0.001,
            # max_grad_norm=0.3
        )

        part_oof = train_function(OUTPUT_DIR, args, fold)
        res.append(part_oof)

    oof = pd.concat(res)
    oof['score'] = oof['score'] + 1
    oof.to_csv(f'{OUTPUT_DIR}/oof.csv', index=None)

    y_true = oof['score']
    y_pred = oof['pred']
    total_score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f'total cv is {total_score}')


if __name__ == '__main__':
    main()

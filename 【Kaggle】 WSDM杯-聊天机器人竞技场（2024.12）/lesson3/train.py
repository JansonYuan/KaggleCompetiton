import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import os
import copy
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    AutoConfig,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
from unsloth import FastLanguageModel


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "/gemma-2-9b-it/"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 1024
    n_splits: int = 5
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2  # global batch size is 8
    per_device_eval_batch_size: int = 32
    n_epochs: int = 1
    freeze_layers: int = 0  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 2e-4
    warmup_steps: int = 0
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"


config = Config()
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=200,
    optim=config.optim_type,
    bf16=True,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
)

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj"],
    modules_to_save=['lm_head'],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.CAUSAL_LM,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"

# model = Gemma2ForSequenceClassification.from_pretrained(
#     config.checkpoint,
#     num_labels=2,
#     #torch_dtype=torch.bfloat16,
#     #device_map="auto",
# )
# model.config.use_cache = False
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)

model_config = AutoConfig.from_pretrained(config.checkpoint)

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit=True
model, _ = FastLanguageModel.from_pretrained(
    model_name = config.checkpoint,
    max_seq_length = config.max_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
model.lm_head = nn.Linear(model_config.hidden_size, 2, bias=False)
model.lm_head = model.lm_head.to(model.device)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)




ds = Dataset.from_parquet('train.parquet')


class CustomTokenizer:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + t for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + t for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + t for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        labels = [0 if i == 'model_a' else 1 for i in batch["winner"]]
        return {**tokenized, "labels": labels}

encode = CustomTokenizer(tokenizer, max_length=config.max_length)
ds = ds.map(encode, batched=True)

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

folds = [
    (
        [i for i in range(len(ds)) if i % config.n_splits != fold_idx],
        [i for i in range(len(ds)) if i % config.n_splits == fold_idx]
    )
    for fold_idx in range(config.n_splits)
]

train_idx, eval_idx = folds[config.fold_idx]

trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds.select(train_idx),
    eval_dataset=ds.select(eval_idx),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
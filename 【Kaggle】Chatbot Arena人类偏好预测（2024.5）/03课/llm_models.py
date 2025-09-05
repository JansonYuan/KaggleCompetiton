import torch
import torch.nn as nn
from loss import WeightedKappaLoss
from transformers import AutoConfig, BitsAndBytesConfig, MistralPreTrainedModel, MistralModel, \
    AutoModelForSequenceClassification
from transformers.utils import add_start_docstrings_to_model_forward
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast


class AESMistral(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        self.in_dim = self.config.hidden_size
        self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1,
                              dropout=0.0, batch_first=True,
                              bidirectional=True)
        self.pool = MeanPooling()
        self.last_fc = nn.Linear(self.in_dim * 2, self.config.num_labels)
        # self.fc = nn.LazyLinear(num_classes)
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)
        # Initialize weights and apply final processing
        self.post_init()
        self.loss_function = nn.CrossEntropyLoss()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        #print(hidden_states)
        x, _ = self.bilstm(hidden_states)
        x = self.pool(x, attention_mask)
        logits = self.last_fc(x)

        loss = None
        output = (logits,)
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )



class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class LLMAESModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_ratio=0.05, use_kbit=False, task_type='cls',
                 token='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', use_kappa_loss=True, use_lora=True):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(
            model_name,
        )

        self.config = AutoConfig.from_pretrained(model_name)
        self.task_type = task_type
        self.num_labels = num_labels
        self.config.num_labels = num_labels
        if task_type != 'cls':
            self.config.hidden_dropout_prob = 0
            self.config.attention_probs_dropout_prob = 0
            self.config.attention_dropout = 0
            self.config.problem_type = "regression"
        self.in_dim = self.config.hidden_size

        self.config.use_cache = True
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            device_map="auto",
            # quantization_config=bnb_config,
            token=token
        )
        # bert_model.gradient_checkpointing_enable()
        self.bert_model.config.pad_token_id = self.bert_model.config.eos_token_id
        peft_config = LoraConfig(
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
            inference_mode=False,
            r=6,
            lora_alpha=32,
            lora_dropout=dropout_ratio,
        )
        if use_lora:
            self.bert_model = get_peft_model(self.bert_model, peft_config)
            #self.model = self.bert_model.merge_and_unload()
        if use_kbit:
            self.bert_model = prepare_model_for_kbit_training(self.bert_model, use_gradient_checkpointing=True)
        self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1,
                              dropout=dropout_ratio, batch_first=True,
                              bidirectional=True)
        self.pool = MeanPooling()
        self.last_fc = nn.Linear(self.in_dim * 2, self.config.num_labels)
        # self.fc = nn.LazyLinear(num_classes)
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)
        if task_type != 'cls':
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()
            if use_kappa_loss:
                self.loss_function = WeightedKappaLoss(num_classes=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        x, _ = self.bilstm(x)
        x = self.pool(x, attention_mask)
        logits = self.last_fc(x)

        loss = None
        if labels is not None:
            if self.task_type != 'cls':
                loss = self.loss_function(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        output = (logits,)
        return ((loss,) + output) if loss is not None else output


if __name__ == '__main__':
    print('')

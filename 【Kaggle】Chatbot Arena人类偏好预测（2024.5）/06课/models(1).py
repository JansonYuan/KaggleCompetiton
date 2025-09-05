import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast


"""

old:
prompt + model_a + model_b ----> BERT|LLM ----> score

new:
prompt + model_a ----> BERT|LLM ----> hidden states 0
                                                      ----> concat ----> FC ----> score
prompt + model_b ----> BERT|LLM ----> hidden states 1


"""
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

class CustomModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(
            model_name,
        )

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.max_position_embeddings = 4096
        self.config.num_labels = 3
        self.in_dim = self.config.hidden_size

        self.bert_model = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        # self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1,
        #                       dropout=self.config.hidden_dropout_prob, batch_first=True,
        #                       bidirectional=True)
        self.pool = MeanPooling()
        self.last_fc = nn.Linear(self.in_dim * 2, self.config.num_labels)
        
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)
        

    def forward(self, 
                input_ids_0, 
                attention_mask_0,
                token_type_ids_0=None,
                input_ids_1=None, 
                attention_mask_1=None,
                token_type_ids_1=None,
                labels=None):
        x_0 = self.bert_model(input_ids_0, attention_mask=attention_mask_0, token_type_ids=token_type_ids_0)[0] # b, s, hidden_size
        #x_0, _ = self.bilstm(x_0)
        x_0 = self.pool(x_0, attention_mask_0) # b, hidden_size

        x_1 = self.bert_model(input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)[0]
        #x_0, _ = self.bilstm(x_0)
        x_1 = self.pool(x_1, attention_mask_1)

        x = torch.cat([x_0, x_1], dim=-1) # b, hidden_size * 2
        logits = self.last_fc(x) # b, 3

        loss = None
        output = (logits,)
        return {'logits':logits, 'loss':loss}
    
class CustomModelRank(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(
            model_name,
        )

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.max_position_embeddings = 4096
        self.config.attention_probs_dropout_prob = 0
        self.config.hidden_dropout_prob = 0
        self.config.num_labels = 1
        self.in_dim = self.config.hidden_size

        self.bert_model = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        # self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1,
        #                       dropout=self.config.hidden_dropout_prob, batch_first=True,
        #                       bidirectional=True)
        self.pool = MeanPooling()
        self.last_fc = nn.Linear(self.in_dim * 2, self.config.num_labels)
        
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)
        

    def forward(self, 
                input_ids_0, 
                attention_mask_0,
                token_type_ids_0=None,
                input_ids_1=None, 
                attention_mask_1=None,
                token_type_ids_1=None,
                labels=None):
        x_0 = self.bert_model(input_ids_0, attention_mask=attention_mask_0, token_type_ids=token_type_ids_0)[0] # b, s, hidden_size
        #x_0, _ = self.bilstm(x_0)
        x_0 = self.pool(x_0, attention_mask_0) # b, hidden_size

        x_1 = self.bert_model(input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)[0]
        #x_0, _ = self.bilstm(x_0)
        x_1 = self.pool(x_1, attention_mask_1)

        x = torch.cat([x_0, x_1], dim=-1) # b, hidden_size * 2
        logits = self.last_fc(x) # b, 3

        loss = None
        output = (logits,)
        return {'logits':logits, 'loss':loss}
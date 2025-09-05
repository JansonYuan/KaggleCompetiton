import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

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

class AESModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(
            model_name,
        )

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.max_position_embeddings = 4096
        self.config.num_labels = 1
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.in_dim = self.config.hidden_size

        self.bert_model = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )

        self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1,
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        self.pool = MeanPooling()
        self.last_fc = nn.Linear(self.in_dim * 2, self.config.num_labels)
        # self.fc = nn.LazyLinear(num_classes)
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)
        self.loss_function = nn.MSELoss()
        self.num_labels = 3
	self.drop_0 = nn.Dropout(0.1)
	self.drop_1 = nn.Dropout(0.2)
	self.drop_2 = nn.Dropout(0.3)


    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if self.if_llm:
	    for i in range(5):
            	x += self.bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1] / 5.0
        else:
            x = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        x, _ = self.bilstm(x)
        x = self.pool(x, attention_mask)
        logits_0 = self.last_fc(self.drop_0(x))
 	logits_1 = self.last_fc(self.drop_1(x))
	logits_2 = self.last_fc(self.drop_2(x))
	logits = ( logits_0 + logits_1 +logits_2 ) / 3.0

        loss = None
        if labels is not None:
            loss = self.loss_function(logits.view(-1), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output

if __name__ == '__main__':

    print('')

from transformers import PreTrainedTokenizerBase
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

answer_map = {0:'model_a', 1:'model_b', 2: 'tie'}
rank_label_map = {0:0, 1:2, 2: 1} #model_a, tie, model_b
class UnifiedSFTDataset(Dataset):
    """
    统一的数据处理dataset
    """
    def __init__(self, instructs, model_a_response, model_b_response, labels, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instructs = instructs
        self.model_a_response = model_a_response
        self.model_b_response = model_b_response
        self.labels = labels #[answer_map[i] for i in labels]


    def __len__(self):
        return len(self.instructs)

    def __getitem__(self, index):
        output = {}
        prompt = self.instructs[index]
        model_a_response = self.model_a_response[index]
        model_b_response = self.model_b_response[index]
        label = self.labels[index]
        tmp = self.tokenizer(prompt, model_a_response, max_length=self.max_seq_length, truncation=True)
        output['input_ids_0'] =  tmp['input_ids']
        output['attention_mask_0'] =  tmp['attention_mask']
        output['token_type_ids_0'] =  tmp['token_type_ids']


        tmp = self.tokenizer(prompt, model_b_response, max_length=self.max_seq_length, truncation=True)
        output['input_ids_1'] =  tmp['input_ids']
        output['attention_mask_1'] =  tmp['attention_mask']
        output['token_type_ids_1'] =  tmp['token_type_ids']

        output['labels'] =  label
        return output


class UnifiedSFTDatasetRank(Dataset):
    """
    统一的数据处理dataset
    """
    def __init__(self, instructs, model_a_response, model_b_response, labels, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instructs = instructs
        self.model_a_response = model_a_response
        self.model_b_response = model_b_response
        self.labels = labels #[answer_map[i] for i in labels]


    def __len__(self):
        return len(self.instructs)

    def __getitem__(self, index):
        output = {}
        prompt = self.instructs[index]
        model_a_response = self.model_a_response[index]
        model_b_response = self.model_b_response[index]
        label = self.labels[index]
        tmp = self.tokenizer(prompt, model_a_response, max_length=self.max_seq_length, truncation=True)
        output['input_ids_0'] =  tmp['input_ids']
        output['attention_mask_0'] =  tmp['attention_mask']
        output['token_type_ids_0'] =  tmp['token_type_ids']


        tmp = self.tokenizer(prompt, model_b_response, max_length=self.max_seq_length, truncation=True)
        output['input_ids_1'] =  tmp['input_ids']
        output['attention_mask_1'] =  tmp['attention_mask']
        output['token_type_ids_1'] =  tmp['token_type_ids']

        output['labels'] =  rank_label_map[label]
        return output

class SelfDataCollatorWithPadding:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids_0 = [torch.tensor(f['input_ids_0']) for f in features]
        attention_mask_0 = [torch.tensor(f['attention_mask_0']) for f in features]
        token_type_ids_0 = [torch.tensor(f['token_type_ids_0']) for f in features]

        input_ids_1 = [torch.tensor(f['input_ids_1']) for f in features]
        attention_mask_1 = [torch.tensor(f['attention_mask_1']) for f in features]
        token_type_ids_1 = [torch.tensor(f['token_type_ids_1']) for f in features]

        labels = torch.as_tensor([torch.tensor(f['labels']) for f in features], dtype=torch.long)

        padded_input_ids_0 = pad_sequence(input_ids_0, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask_0 = pad_sequence(attention_mask_0, batch_first=True, padding_value=0)
        padded_token_type_ids_0 = pad_sequence(token_type_ids_0, batch_first=True, padding_value=0)

        padded_input_ids_1 = pad_sequence(input_ids_1, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask_1 = pad_sequence(attention_mask_1, batch_first=True, padding_value=0)
        padded_token_type_ids_1 = pad_sequence(token_type_ids_1, batch_first=True, padding_value=0)



        return {
            'input_ids_0': padded_input_ids_0,
            'attention_mask_0': padded_attention_mask_0,
            'token_type_ids_0': padded_token_type_ids_0,
            'input_ids_1': padded_input_ids_1,
            'attention_mask_1': padded_attention_mask_1,
            'token_type_ids_1': padded_token_type_ids_1,
            'labels': labels
        }

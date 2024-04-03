import torch
import torch.nn.functional as F
from transformers import AutoModel


# init model
class CustomModel(torch.nn.Module):
    def __init__(self, n_class=4):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained("vinai/bertweet-large")
        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, n_class)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        pred = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.linear1(pred["last_hidden_state"][:,0,:].view(-1,1024))
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = F.log_softmax(output, dim=1)
        output = torch.exp(output)
        return output
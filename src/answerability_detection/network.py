import torch.nn as nn 

class AnsModelV1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AnsModelV1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
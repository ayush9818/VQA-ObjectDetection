import torch 
import torch.nn as nn
import torch.nn.functional as F

class VQAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.dropout2(x)
        return x


class VQAModelV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModelV2, self).__init__()

        self.block1 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)])

        self.block2 = nn.Sequential(*[nn.Linear(hidden_dim, 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)])

        self.block3 = nn.Sequential(*[nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)])

        self.classifier = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.classifier(x)
        return out

class VQAModelV3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModelV3, self).__init__()

        self.block1 = nn.Sequential(*[nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)])

        self.block2 = nn.Sequential(*[nn.Linear(hidden_dim, 2*hidden_dim),
            nn.LayerNorm(2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)])

        self.classifier = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        out = self.classifier(x)
        return out
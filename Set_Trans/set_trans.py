import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from modules import SAB, PMA
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd

num_classes = 4


class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x


class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=2048, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)



# Dataset class definition
class SetDataset(Dataset):
    def __init__(self, data_folder, annotations_file, mode='train'):
        self.data_folder = data_folder
        self.annotations = pd.read_csv(annotations_file)
        self.mode = mode
        self.annotations['path'] = self.annotations['filename'] + '_' + self.annotations['x_y'] + '.pt'
        self.annotations.set_index('path', inplace=True)
        self.data_files = [f for f in os.listdir(data_folder) if f.endswith('.pt')]
        if mode == 'train':
            self.data_files = [f for f in self.data_files if not any(x in f for x in ['case3', 'case4', 'control3', 'control4'])]
        else:
            self.data_files = [f for f in self.data_files if any(x in f for x in ['case3', 'case4', 'control3', 'control4'])]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_path = self.data_files[index]
        data = torch.load(os.path.join(self.data_folder, file_path))
        label = self.annotations.loc[file_path, 'level']
        label = torch.tensor(label, dtype=torch.long)
        return data, label

# Collate function to handle variable-sized data
def collate_fn(batch):
    xs, ys = zip(*batch)
    return list(xs), torch.tensor(ys, dtype=torch.long)


def train(model, data_loader, epochs, save_path):
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)  # Using CrossEntropyLoss as it is common for classification tasks
    losses = []

    # Set model to training mode
    model.train()
    
    for epoch in range(epochs):
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), save_path)
    return losses

def evaluate(model, data_loader, load_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the saved weights
    model.load_state_dict(torch.load(load_path))

    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss.item()
            predictions = output.argmax(dim=1, keepdim=True)
            total_correct += predictions.eq(label.view_as(predictions)).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


data_folder = '/projectnb/ec500kb/projects/project7/GTEx/annotated_patches/resnet_features' # the path for the stored features
annotations_file = '/projectnb/ec500kb/projects/project7/GTEx/annotated_patches/annotations.csv'
train_dataset = SetDataset(data_folder, annotations_file, mode='train')
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)

test_dataset = SetDataset(data_folder, annotations_file, mode='test')
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)

# Assuming you have a Set Transformer model defined as 'model'
model = SmallSetTransformer()

train_path = '/projectnb/ec500kb/projects/project7/set_transformer/model_weights.pth'
evaluate_path = '/projectnb/ec500kb/projects/project7/set_transformer/model_weights.pth'

train(model, train_loader, epochs=50, save_path=train_path)
evaluate(model, test_loader, load_path=evaluate_path)



# def infer(model, data_loader):
#     model.eval()  # Set the model to evaluation mode
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     predictions = []

#     with torch.no_grad():  # No gradients needed during inference
#         for data in data_loader:
#             data = data.to(device)
#             output = model(data)
#             # Convert output probabilities to predicted class (max probability)
#             pred_classes = output.argmax(dim=1)
#             predictions.extend(pred_classes.cpu().numpy())

#     return predictions

# # Example of setting up data loader for inference (assuming no labels)
# class InferenceDataset(Dataset):
#     """Dataset for loading data without labels for inference"""
#     def __init__(self, data_folder):
#         self.data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pt')]

#     def __len__(self):
#         return len(self.data_files)

#     def __getitem__(self, index):
#         data = torch.load(self.data_files[index])
#         return data

# # Setup inference dataset and loader
# inference_data_folder = 'path_to_your_inference_data_folder'
# inference_dataset = InferenceDataset(inference_data_folder)
# inference_loader = DataLoader(inference_dataset, batch_size=20, shuffle=False)

# # Perform inference
# model_predictions = infer(model, inference_loader)

from __future__ import absolute_import, division, print_function, unicode_literals

# Pytorch and Pytorch Geometric
import torch as tch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import GCNConv, summary as gsummary, global_mean_pool
from torch_geometric.data import Data, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

# Helper libraries
from torchsummary import summary as asummary
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")


def annFit(ann_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs, optimizer, criterion):
    train_losses = []
    val_losses = []
    ann_model.to(device)
    pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in range(num_epochs):
        ann_model.train()
        optimizer.zero_grad()
        outputs = ann_model(X_train_tensor.to(device))
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        ann_model.eval()
        with tch.no_grad():
            val_outputs = ann_model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_losses.append(val_loss.item())
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        pbar.update(1)
        pbar.set_postfix({"Training Loss": train_losses[-1], "Validation Loss": val_losses[-1]})
    return ann_model, train_losses, val_losses

def fitGNN(gnn_model, t_loader, v_loader, num_epochs, batch_size, optimizer, criterion):
    train_losses = []
    val_losses = []
    pbar = tqdm(range(num_epochs), desc="Epochs")
    pbar_t = tqdm(total=len(t_loader), desc="Training Batch:", leave=False)
    pbar_v = tqdm(total=len(v_loader), desc="validation Batch:", leave=False)
    for epoch in range(num_epochs):
        # Training Phase
        gnn_model.train()
        train_loss_items = []
        pbar_t.reset()
        pbar_v.reset()
        for batch in t_loader:
            optimizer.zero_grad()
            # Use Batch Data object in forward pass
            outputs = gnn_model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            train_loss_items.append(loss.item())
            pbar_t.update()
        avg_train_loss = sum(train_loss_items) / len(train_loss_items)
        train_losses.append(avg_train_loss)
        # Validation Phase (assuming you have a separate validation loader)
        gnn_model.eval()
        val_loss_items = []
        with tch.no_grad():
            for val_batch in v_loader:
                val_outputs = gnn_model(val_batch.x, val_batch.edge_index, val_batch.batch)
                val_loss = criterion(val_outputs, val_batch.y)
                val_loss_items.append(val_loss.item())
                pbar_v.update()

        avg_val_loss = sum(val_loss_items) / len(val_loss_items)
        val_losses.append(avg_val_loss)
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {avg_val_loss:.4f}')
        pbar.update(1)
        pbar.set_postfix({"Training Loss": avg_train_loss, "Validation Loss": avg_val_loss})
    return gnn_model, train_losses, val_losses

def plot_history(train_losses, val_losses, model_name):
    fig = plt.figure(figsize=(15, 5), facecolor='w')
    ax = fig.add_subplot(121)
    ax.plot(train_losses)
    ax.plot(val_losses)
    ax.set(title=model_name + ': Model loss', ylabel='Loss', xlabel='Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')
    ax = fig.add_subplot(122)
    ax.plot(np.log(train_losses))
    ax.plot(np.log(val_losses))
    ax.set(title=model_name + ': Log model loss', ylabel='Log loss', xlabel='Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    plt.close()



def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
    return list(fp.ToBitString())

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def read_smiles_data(path_data):
    df = pd.read_csv(path_data, sep=',')
    df['fingerprint'] = df['SMILES'].apply(smiles_to_fingerprint)
    df['fingerprint'] = df['fingerprint'].apply(lambda x: [int(bit) for bit in x])
    df['mol'] = df['SMILES'].apply(smiles_to_mol)
    return df

def make_pyg(row):
    # Create node features
    mol = row['mol']
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x1 = tch.tensor(atom_features, dtype=tch.float).view(-1, 1)
    x2 = tch.tensor(row['fingerprint'], dtype=tch.float)
    y = tch.tensor(row['measured.log.solubility.mol.L.'], dtype=tch.float).view(-1, 1)
    # Duplicate fingerprint for each atom and concatenate
    x2_repeated = x2.repeat(x1.shape[0], 1)
    x = tch.cat([x1, x2_repeated], dim=1)
    
    # Create edge features (connectivity)
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_features.append([bond.GetBondTypeAsDouble()])  # Bond type (as double)
    
    edge_index = tch.tensor(edge_indices, dtype=tch.long).t().contiguous()
    edge_attr = tch.tensor(edge_features, dtype=tch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer with 1 node
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class GNN(nn.Module):
    def __init__(self, input_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc3 = nn.Linear(16, 1)  # Output layer with 1 node
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc3(x)
        return x
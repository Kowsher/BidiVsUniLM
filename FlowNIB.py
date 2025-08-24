import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
from sklearn.decomposition import PCA

deff_list = []
    
lists=[]
print('train_data', len(train_data['Z']))
for i in tqdm(range(0,  len(train_data['Z']))): # for every layer here train_data['Z'] -> (layer, batch, dim)
    ind = i
    X_train = train_data['X']
    Z_train = train_data['Z'][ind]
    Y_train = train_data['Y']#.reshape(-1, 1)   # Using Y from dataset


    # embeddings: (n_samples, d) numpy array
    pca = PCA()
    pca.fit(Z_train)

    lambdas = pca.explained_variance_  # eigenvalues (variances)

    d_eff = (np.sum(lambdas))**2 / np.sum(lambdas**2) 
    deff_list.append(d_eff)

    print(f"Effective Dimension d_eff = {d_eff:.2f}")

    pca = PCA()
    pca.fit(Y_train)

    lambdas = pca.explained_variance_  # eigenvalues (variances)

    d_eff_y = (np.sum(lambdas))**2 / np.sum(lambdas**2) 

    print(f"Effective Dimension d_eff of y = {d_eff_y:.2f}")


    if not torch.is_tensor(Y_train):
        Y_train = torch.tensor(Y_train, dtype=torch.float32 if len(Y_train.shape) == 1 else torch.long)
       

    print("Train shapes:", X_train.shape, Z_train.shape, Y_train.shape)


    # ================================
    # 2. Define Models
    # ================================

    class Encoder(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Encoder, self).__init__()
            self.fc = nn.Linear(input_size, hidden_size)
            self.activation = nn.ReLU()

        def forward(self, x):
            z = self.activation(self.fc(x))
            return z

    class MINE(nn.Module):
        def __init__(self, size1, size2):
            super(MINE, self).__init__()
            self.fc1 = nn.Linear(size1 + size2, 512)
            #self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 1)
            self.act = nn.Tanh()

        def forward(self, a, b):
            # Make sure both are 2D
            if a.dim() == 1:
                a = a.unsqueeze(1)
            if b.dim() == 1:
                b = b.unsqueeze(1)
            ab = torch.cat([a, b], dim=1)
            h = self.act(self.fc1(ab))
            #h = self.act(self.fc2(h))
            out = self.fc3(h)
            return out

    class MaximizeMINE(nn.Module):
        def __init__(self, input_size, hidden_size, label_size):
            super(MaximizeMINE, self).__init__()
            self.encoder = Encoder(input_size, hidden_size)
            self.mine_xz = MINE(input_size, hidden_size)
            self.mine_zy = MINE(hidden_size, label_size)
            

        def compute_mine_loss(self, mine_net, a, b):
            # Positive samples
            T_pos = mine_net(a, b)
            # Negative samples (shuffled)
            b_shuffled = b[torch.randperm(b.size(0))]
            T_neg = mine_net(a, b_shuffled)

            # Donsker-Varadhan MI estimation
            max_val = torch.max(T_neg)
            log_sum_exp = max_val + torch.log(torch.exp(T_neg - max_val).mean() + 1e-6)

            mi_estimate = (T_pos.mean() - log_sum_exp)
            return mi_estimate

        def forward(self, x, y, step):
            z = self.encoder(x)

            I_xz = self.compute_mine_loss(self.mine_xz, x, z)/(d_eff)**2
            I_zy = self.compute_mine_loss(self.mine_zy, z, y)/(d_eff_y)**2


            # ✅ Maximize weighted sum of MI
            loss = -( step*I_xz + (1-step)*I_zy )

            return loss, I_xz*(d_eff)**2, I_zy*(d_eff_y)**2, step

    # ================================
    # 3. Training Function
    # ================================

    def train_maximize_mine(model, train_loader, epochs=10000, lr=1e-3, max_grad_norm=5.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        Ixz_list = []
        Iyz_list = []
        step=1

        for epoch in range(epochs):
            
            model.train()
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                loss, I_xz, I_zy, step = model(x_batch, y_batch, step)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Validation every 10 epochs
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, I(X,Z) = {I_xz.item():.4f}, I(Y,Z) = {I_zy.item():.4f}")
                #print(f"           Val   Loss = {val_loss.item():.4f}, I(X,Z) = {val_I_xz.item():.4f}, I(Y,Z) = {val_I_yz.item():.4f}")
            step = step-0.001
            step = max(0, step)
            #print('step ', step)
            Ixz_list.append(I_xz.item())
            Iyz_list.append(I_zy.item())

        return Ixz_list, Iyz_list

    # ================================
    # 4. Dataset and Training
    # ================================

    batch_size = 1000

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



    input_size = X_train.shape[1]
    hidden_size = Z_train.shape[1]
    label_size = 1 if Y_train.dim() == 1 else Y_train.shape[1]

    model = MaximizeMINE(input_size=input_size, hidden_size=hidden_size, label_size=label_size)

    Ixz_list, Iyz_list = train_maximize_mine(model, train_loader, epochs=2000)
    lists.append((Ixz_list, Iyz_list))
   
    del model

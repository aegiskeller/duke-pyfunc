# Write code to construct a variational autoencoder (VAE) model using 
# PyTorch. The VAE model should an input set that consists of 784 
# synthetically generated trajectories of 20 points each. The trajectories 
# will have four distinct behaviours: straight, left turn, right turn, and
# random. The VAE model should have an encoder and a decoder. The encoder
# should have a single hidden layer with 100 units and the decoder should
# have a single hidden layer with 100 units. The latent space should have
# 2 dimensions. The VAE model should be trained using the Adam optimizer
# with a learning rate of 0.001. The loss function should be the negative
# log likelihood loss. The VAE model should be trained for 100 epochs. The
# VAE model should be able to generate new trajectories that are similar to
# the input set. The VAE model should be able to generate new trajectories
# that are similar to the input set. 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import Standard
from sklearn.model_selection import train_test_split

# generate synthetic data
np.random.seed(0)
n_samples = 784
n_points = 20
n_features = 2
n_classes = 4

X = np.zeros((n_samples, n_points, n_features))
y = np.random.randint(0, n_classes, n_samples)

for i in range(n_samples):
    if y[i] == 0:
        X[i] = np.cumsum(np.random.randn(n_points, n_features), axis=0)
    elif y[i] == 1:
        X[i] = np.cumsum(np.random.randn(n_points, n_features) + 0.1, axis=0)
    elif y[i] == 2:
        X[i] = np.cumsum(np.random.randn(n_points, n_features) - 0.1, axis=0)
    else:
        X[i] = np.cumsum(np.random.randn(n_points, n_features) * 0.1, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(-1, n_points, n_features)
X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(-1, n_points, n_features)

# convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# create a DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_points * n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 2 * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, n_points * n_features)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, n_points * n_features)
        z = self.encoder(x)
        mu, log_var = z[:, :2], z[:, 2:]
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z)
        x = x.view(-1, n_points, n_features)
        return x, mu, log_var
    
# instantiate the VAE model
vae = VAE()

# define the optimizer and loss function
optimizer = optim.Adam(vae.parameters(), lr=0.001)
criterion = nn.MSELoss()

# train the VAE model
n_epochs = 100  # number of epochs
for epoch in range(n_epochs):
    vae.train()
    train_loss = 0
    for x, _ in train_loader:
        optimizer.zero_grad()
        x_hat, mu, log_var = vae(x)
        loss = criterion(x_hat, x)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss += kl_div
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {train_loss/len(train_loader):.4f}')

# generate new trajectories
vae.eval()
with torch.no_grad():
    z = torch.randn(1, 2)
    x_hat = vae.decoder(z)
    x_hat = x_hat.view(n_points, n_features).numpy()
    x_hat = scaler.inverse_transform(x_hat)

# plot the generated trajectory
plt.figure()
plt.plot(x_hat[:, 0], x_hat[:, 1], marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Trajectory')   
plt.show()

# save the model
torch.save(vae.state_dict(), 'vae_model.pth')

# load the model
vae = VAE()
vae.load_state_dict(torch.load('vae_model.pth'))
vae.eval()
print(vae)

# generate new trajectories
vae.eval()
with torch.no_grad():
    z = torch.randn(1, 2)
    x_hat = vae.decoder(z)
    x_hat = x_hat.view(n_points, n_features).numpy()
    x_hat = scaler.inverse_transform(x_hat)

# plot the generated trajectory
plt.figure()    
plt.plot(x_hat[:, 0], x_hat[:, 1], marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Trajectory')
plt.show()




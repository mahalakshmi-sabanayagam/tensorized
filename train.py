import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import kmeans_plusplus
from models import TensorisedAEloss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_AE(net, X, Y, lr=0.1, epochs=100, CNN=False, X_out=None):
    if X_out == None:
        X_out = X

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.1)
    X = X.to(device)
    Y = Y.to(device)
    X_out = X_out.to(device)
    train_loss = []
    print(X.type())
    for epoch in range(epochs):
        total_loss = 0
        for i in range(X.shape[0]):
            x = X[i]
            x_out = X_out[i]
            if CNN:
                # this can def be optimized
                x = x.reshape(28, 28)
                x = torch.unsqueeze(x, dim=0)
                x = torch.unsqueeze(x, dim=0)

                x_out = x_out.reshape(28, 28)
                x_out = torch.unsqueeze(x_out, dim=0)
                x_out = torch.unsqueeze(x_out, dim=0)

            optimizer.zero_grad()
            out = net(x.float())
            loss = criterion(out, x_out.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss = total_loss / X.shape[0]
        if (epoch + 1) % 50 == 0:
            print('epoch ', epoch, ' loss ', loss)
        train_loss.append(loss)

    return train_loss


def train_TAE(X, Y, n_clusters=2, lr=0.1, reg=0, embed=2, epochs=100, number_of_batches=1, linear=True, CNN=False, X_out=None):
    if X_out == None:
        X_out = X
    X = X.to(device)
    Y = Y.to(device)
    X_out = X_out.to(device)
    train_loss = []

    centers, indices = kmeans_plusplus(X_out.cpu().detach().numpy(), n_clusters=n_clusters, random_state=20)

    clust = None
    for i in range(n_clusters):
        d = torch.norm(X_out - torch.tensor(centers[i]).to(device), dim=1).reshape(-1, 1)
        if clust is None:
            clust = d
        else:
            clust = torch.cat((clust, d), dim=1).to(device)
    clust = torch.argmin(clust, axis=1).to(device)
    clust_assign = torch.zeros([n_clusters, X.shape[0]], dtype=torch.float64).to(device)
    for i in range(X.shape[0]):
        clust_assign[clust[i], i] = 1

    centers = clust_assign.float() @ X_out.float()
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1],
                                                                                          dtype=torch.float).to(device)
    centers = centers / norm

    net = TensorisedAEloss(X.shape[1], embed, reg=reg, num_clusters=n_clusters, linear=linear, CNN=CNN).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.1)

    for epoch in range(epochs):
        # x_shuffle = torch.randperm(X.shape[0])
        # X = X[x_shuffle]
        # Y = Y[x_shuffle]
        # clust_assign = clust_assign[:, x_shuffle]

        total_loss = 0
        batch_size = int(X.shape[0] / number_of_batches)
        for b in range(int(X.shape[0] / batch_size)):
            optimizer.zero_grad()
            temp_idx = []
            batch_loss = 0

            # get the loss for the batch and update
            for i in range(batch_size):
                j = b * batch_size + i
                loss_sample, idx = net(X[j].float(), centers, j, clust_assign, X_out[j].float())
                batch_loss += loss_sample
                temp_idx.append(idx)
                total_loss += loss_sample.item()

            batch_loss = batch_loss / batch_size
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            for k in range(batch_size):
                kb = b * batch_size + k
                clust_assign[:, kb] = 0
                clust_assign[temp_idx[k]][kb] = 1

            new_centers = clust_assign.float() @ X_out.float()
            new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1],
                                                                                                      dtype=torch.float).to(
                device)
            new_centers = new_centers / new_norm
            centers = (b * centers + new_centers) / (b + 1)

        loss = total_loss / X.shape[0]
        if (epoch + 1) % 50 == 0:
            print('epoch ', epoch, ' loss ', loss)
        train_loss.append(loss)

    return net, train_loss, clust_assign, X, Y, X_out
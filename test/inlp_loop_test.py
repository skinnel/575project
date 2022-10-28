""" A script to test out the INLP loop and ensure the projection intersection is calculated correctly """

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy
import random

from typing import List
from numpy.linalg import matrix_rank

# Wanted to switch around some of the defaults here, so copied for now instead of importing
# from src.LinearClassifier import LinearClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_embeddings, output, tag_size):
        '''
        a linear classifier for probe
        input_embeddings : a tensor with size [batch_size,embed_dim]
        output : a tensor with size [batch_size]
        tag_size : number of classes
        dev_x: dev set for stopping criterion
        dev_y: dev label for stopping criterion
        '''
        super().__init__()
        random.seed(42)
        ## everything defined in GPU
        self.embeddings = input_embeddings
        self.output = output
        self.linear = torch.nn.Linear(input_embeddings.shape[1], tag_size, device=device, dtype=torch.double)
        # class weight performs really worse
        # cls_weight = compute_class_weight('balanced',classes=np.array(range(tag_size)),y=output.numpy())
        # cls_weight = torch.tensor(cls_weight,dtype=torch.float)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings):
        # embedding size = [batch_size, embed_dim]
        # output size = [batch_size,tag_size]
        emb = embeddings.to(device)
        emb = emb.double()
        fc = self.linear(emb)
        return fc

    def eval(self, dev_x, dev_y):
        with torch.no_grad():
            dev_x = dev_x.to(device)
            dev_y = dev_y.to(device)
            dev_pred = self.forward(dev_x)
            loss = self.loss_func(dev_pred, dev_y)

        final_dev = torch.argmax(dev_pred, dim=1).cpu().numpy()
        print(f'dev accuracy score:{accuracy_score(dev_y.cpu().numpy(), final_dev):.4f}')
        return dev_pred, loss.item()

    def batched_input(self, *args, batch_size=64):
        data_set = TensorDataset(args[0], args[1])
        dataloader = DataLoader(data_set, batch_size=batch_size)
        return dataloader

    def optimize(self, lr=0.001, num_epochs=500):
        optimizer = torch.optim.AdamW(self.linear.parameters(), lr=lr)
        best_predictions = None
        best_loss = float('inf')
        stop_count = 0
        output = self.output.to(device)
        dataloader = self.batched_input(self.embeddings, output)
        for epoch in range(num_epochs):
            preds = []
            total_loss = 0
            for emb, label in dataloader:
                optimizer.zero_grad()
                pred = self.forward(emb)
                loss = self.loss_func(pred, label)
                loss.backward(retain_graph=True)
                optimizer.step()
                pred = pred.to('cpu')
                preds.append(pred)
                total_loss += loss.item()

            total_loss = total_loss / len(dataloader)
            preds = torch.cat(preds)
            # print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
            # implement stopping criterion
            if total_loss < best_loss:
                best_loss = total_loss
                best_model = self.linear
                best_predictions = preds
                stop_count = 0
            else:
                if stop_count == 5:
                    break
                else:
                    stop_count += 1
        final_pred = torch.argmax(best_predictions, dim=1).cpu().numpy()
        # final_dev = torch.argmax(best_dev,dim=1).numpy()
        # final_out = output.numpy()
        # dev_out = self.dev_y.numpy()
        train_acc = accuracy_score(self.output.numpy(), final_pred)
        print(f'train accuracy score:{train_acc:.4f}')
        # print(f'dev accuracy score:{accuracy_score(self.dev_y.numpy(),final_dev):.4f}')

        return best_model, train_acc


# Functions adapted from INLPTraining in LinearClassifier.py
def get_rowspace_projection(W):
    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis
    w_basis = w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace
    return P_W


def get_projection_to_intersection_of_nullspaces(input_dim: int,
        rowspace_projection_matrices: List[np.ndarray]):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param input_dim: input dim
    """
    # This is werid because Q is not normalized so the N(P) = I-P does not work
    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)
    return P


def reinitialize_classifier(in_size, out_size):
    ## may be empty cache here
    random.seed(42)
    linear_model = torch.nn.Linear(in_size, out_size, device=device, dtype=torch.double)
    return linear_model


def apply_projection(original_embedding, P):
    '''
    applying projection of P to the embedding vectors
    '''
    ## may be empty cache here
    P = torch.tensor(P, dtype=torch.double)
    embeddings = torch.matmul(P, original_embedding.T)
    embeddings = embeddings.T.double()
    return embeddings


# Initialize test data
N = 1000
d = 300
X0 = np.random.rand(N, d) - 0.5
X0 = torch.tensor(X0)
X = X0
Y = np.array([1 if sum(x) > 0 else 0 for x in X])  # X < 0 #np.random.rand(N) < 0.5 #(X + 0.01 * (np.random.rand(*X.shape) - 0.5)) < 0 #np.random.rand(5000) < 0.5
Y = torch.tensor(Y)

input_dim = X.shape[1]
n_classes = 2
iteration = 500
min_acc = Y.sum().item() / Y.shape[0] # This only works for 2 classes

# Track matrix ranks throughout loop
p_rank_hx = [0]
emb_rank_hx = [matrix_rank(X)]
x1_rank_hx = [matrix_rank(X)]

I = np.eye(input_dim)
P = I
Ws = []
all_P = []
rowspace_projections = []

# This should really be a while loop that stops when we pass the accuracy threshold
i = 0
acc = 100*min_acc
P_Nwi = np.eye(X.shape[1])
while acc > min_acc :

    #linear_model = reinitialize_classifier(in_size=d, out_size=n_classes)
    linear_model = LinearClassifier(input_embeddings=X, output=Y, tag_size=n_classes)
    bm, acc = linear_model.optimize()
    W = bm.weight.detach().cpu().numpy()
    Ws.append(W)

    # Noted this is the projection space for W, not the null space
    P_rowspace_wi = get_rowspace_projection(W)
    rowspace_projections.append(P_rowspace_wi)

    # This line is supposed to get the null space for the projection space of W
    # Intuitively I think the rank makes sense, but I don't know how this will hold
    #P_Nwi = I - P_rowspace_wi

    # Reverting back to their original code
    P_Nwi = get_projection_to_intersection_of_nullspaces(input_dim=P_rowspace_wi.shape[0],
                                                         rowspace_projection_matrices=rowspace_projections)

    # This line is what they showed originally but the function looks weird
    # P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections)
    P = np.matmul(P_Nwi, P)
    all_P.append(P)
    X = apply_projection(X0,P)

    # For testing purposes: should be same as emb_rank_hx
    X1 = apply_projection(X,P_Nwi)
    x1_rank_hx.append(matrix_rank(X1))

    # Record rank
    p_rank = matrix_rank(P)
    p_rank_hx.append(p_rank)
    x_rank = matrix_rank(X)
    emb_rank_hx.append(x_rank)

    # Reset the linear model parameter cache
    linear_model.linear.reset_parameters()

    i += 1
    print(f'{i}. Accuracy is {acc}')

print(p_rank_hx)
print(emb_rank_hx)
print(x1_rank_hx)

# Quick check to ensure X1 and X are the same
X1[:3,:3]
X[:3,:3]


# Lindsay's Notes:
    # I suspect training for 10 epochs was not enough, should be something like 500. Ideally we'd train long enough for
        #early stopping to be triggered so we can feel confident the model has converged
    # Nullspace projection calculation appeared to be wrong, reverted to code in the gh repo
    # min_accuracy was set to 0, should be percent of majority class in training output
    # should use a while loop that compares acc and min_acc, not pre-set number of iterations
    # tried replacing the rowspace projection calculation step with P_Nwi -= P_rowspace_wi but didn't work as well, so
        # keeping the original code from the gh repo
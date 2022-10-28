""" A module to run the INLP loop """

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy
import random

from typing import List
from numpy.linalg import matrix_rank

from src2.linear_classifier import LinearClassifier

# Helper functions to run the INLP loop
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


#Function to run the INLP loop

def run_inlp(X0: torch.tensor,
             Y: torch.tensor,
             min_acc: Optional(float) = None) -> dict:
    """
    Function to run the INLP loop on the provided data.

    Parameters
    ----------
    X0
        The embeddings to be run through the INLP loop. In the form [embedding_number, embedding_features]
    Y
        The categories to be guarded against that are associated with each embedding.
    min_acc
        The provided minimum accuracy, which is used to determine when the INLP loop should terminate. If not provided
        then we assume the minimum accuracy is equivalent to the accuracy of guessing the majority class for each
        embedding.

    Returns
    -------
    proj_dict
        Dictionary of the final projection matrix, final embeddings, projection matrix history, and embedding history
        throughout the INLP loop
    """


    X = X0
    input_dim = X.shape[1]
    if not min_acc:
        #min_acc = Y.sum().item() / Y.shape[0] # This only works for 2 classes
        _, counts = Y.unique(return_counts=True)[1]
        ct = counts.max()
        n_classes = counts.shape[0]
        min_acc = ct / Y.shape[0]

    # Track matrix ranks throughout loop
    p_rank_hx = [0]
    emb_rank_hx = [matrix_rank(X)]

    I = np.eye(input_dim)
    P = I
    Ws = []
    all_P = []
    rowspace_projections = []

    # This should really be a while loop that stops when we pass the accuracy threshold
    i = 0
    acc = 100*min_acc
    while acc > min_acc :

        # Initialize and fit the linear model
        linear_model = LinearClassifier(input_embeddings=X, output=Y, tag_size=n_classes)
        bm, acc = linear_model.optimize()
        W = bm.weight.detach().cpu().numpy()
        Ws.append(W)

        # Calculate the projection space for W, not the null space
        P_rowspace_wi = get_rowspace_projection(W)
        rowspace_projections.append(P_rowspace_wi)

        # Get the null space of W
        P_Nwi = get_projection_to_intersection_of_nullspaces(input_dim=P_rowspace_wi.shape[0],
                                                             rowspace_projection_matrices=rowspace_projections)

        # Project the embeddings onto the null space of W
        P = np.matmul(P_Nwi, P)
        all_P.append(P)
        X = apply_projection(X0,P)

        # Record rank
        p_rank = matrix_rank(P)
        p_rank_hx.append(p_rank)
        x_rank = matrix_rank(X)
        emb_rank_hx.append(x_rank)

        # Reset the linear model parameter cache
        linear_model.linear.reset_parameters()

        i += 1
        print(f'{i}. Accuracy is {acc}')

    proj_dict = {'new_embeddings': X,
                 'proj_matrix': P,
                 'embedding_rank_hx': emb_rank_hx,
                 'proj_hx': all_P}

    return proj_dict

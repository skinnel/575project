""" A module to instantiate a linear classifier """

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy
import random

from typing import List
from numpy.linalg import matrix_rank


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

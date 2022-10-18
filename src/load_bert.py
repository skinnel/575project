from re import M
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys
import logging
import pandas as pd
import itertools
import pdb
import json
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List
from utils import bert_tokenization
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from nltk.stem import WordNetLemmatizer
#logging.basicConfig(level-logging.INFO) #turn on detailed logging

## defining GPU here
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class DataProcessing:
    '''
    A class for processing data and get right format of input output for model
    '''
    def __init__(self, standard,dset):
        '''
        standard = {gold,silver}
        dset = {train,test,dev}
        '''
        self.standard = standard
        self.dset = dset
        path = '../data/pmb_'+standard+'/'+standard+'_'+dset+'.pkl'
        self.input_df = pd.read_pickle(path)
    
    def bert_tokenize(self,max_len=32):
        '''
        Returns Bert tokenized objects
        '''
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
        input_text = self.input_df['text'].tolist()
        tokens = tokenizer(input_text,
                    max_length=max_len,
                    truncation=True,
                    is_split_into_words=True,
                    padding=True,
                    return_tensors='pt')
        self.tokens = tokens
        return self.tokens

    def get_layerwise_embeddings(self, layer, load_option='save'):
        """
            basically the same thing as get_bert_embedding, except returning the hidden states
        """
        load_path = '../data/pmb_'+self.standard+'/'+self.standard+'_'+self.dset+'_'+layer+'_emb.pt'

        if load_option == 'load':
            output_tensor = torch.load(load_path)
        else:
            input_seqs = self.tokens['input_ids']
            input_mask = self.tokens['attention_mask']
            model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            model = model.to(device)

            ## batching data
            data = TensorDataset(input_seqs, input_mask)
            loader = DataLoader(data, batch_size=1000)

            outputs = []
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

            with torch.no_grad():
                for word_ids, mask in loader:
                    word_ids = word_ids.to(device)
                    mask = mask.to(device)
                    output = model(word_ids, attention_mask=mask)
                    print(len(output))  # check if the hidden states are returned
                    hidden_states = output[2]
                    print(len(hidden_states))  # check the # of hidden layers. Should be 13
                    hidden_states_torch = torch.stack(hidden_states)  # shape (13, 1000, sequence_length, hidden_size)
                    print(hidden_states_torch.shape)  # confirm tensor shape
                    outputs.append(hidden_states_torch)

            output_tensor = torch.cat(outputs, dim=1).detach().cpu()  # concatenating the tensors along the batch_size
            print(output_tensor.shape)  # confirm tensor shape
            torch.save(output_tensor, load_path)

        return output_tensor

    
    def get_bert_embeddings(self,load_option='save'):

        load_path = '../data/pmb_'+self.standard+'/'+self.standard+'_'+self.dset+'_emb.pt'

        if load_option == 'load':
            output_tensor = torch.load(load_path)
            #output_tensor = output_tensor.to(device)
        
        else:
            input_seqs = self.tokens['input_ids']
            input_mask = self.tokens['attention_mask']
            model = AutoModel.from_pretrained('bert-base-uncased')
            model = model.to(device)

            ## batching data
            data = TensorDataset(input_seqs,input_mask)
            loader = DataLoader(data,batch_size=1000)

            outputs = []
            for param in model.parameters():
                param.requires_grad=False
            model.eval()

            with torch.no_grad():
                for word_ids, mask in loader:
                    word_ids = word_ids.to(device)
                    mask = mask.to(device)
                    output = model(word_ids,attention_mask=mask)
                    outputs.append(output[0])
            
            output_tensor = torch.cat(outputs).detach().cpu()
            if load_option == 'save':
                torch.save(output_tensor,load_path)
        
        return output_tensor

    def from_sents_to_words(self,task_keyword,output):
        '''
        this function checks the tokenization process and transfer data into word level
        Arguments:
            task_keyword: {syn,sem}
            output: a tensor with shape (sentence num, max_seq_len, embedding dim)
        Returns: 
            embeddings: a tensor with shape (word num, embedding dim)
            y: a tensor with shape (word num,1)
        '''
        input_text = self.input_df['text'].tolist()
        if task_keyword == 'syn':
            input_y = self.input_df['ccg_tags'].tolist()
        else:
            input_y = self.input_df['semantics_tags'].tolist()
        
        with open('../data/pmb_'+self.standard+'/'+task_keyword+'_mapping.json') as f:
                label_encoder = json.load(f)

        ## mapping tokens back to word ids
        word_inds = [self.tokens.word_ids(i) for i in range(len(input_y))]
        d_list = []
        embed_by_sent = []
        skip_ind = []
        for i in range(len(word_inds)):
            d = defaultdict(list)
            for ind,emb in zip(word_inds[i], output[i]):
                if isinstance(ind, int):
                    d[ind].append(emb)
        
            word_in_sent = [torch.mean(torch.stack(d[k],0),0) if len(d[k])>1 else d[k][0] for k in d.keys()]
            if len(word_in_sent)!= len(input_text[i]):
                skip_ind.append(i)
                continue
            embed_by_sent.append(torch.stack(word_in_sent,0))
            d_list.append(d)
        embeddings = torch.cat(embed_by_sent)
        y = torch.tensor([label_encoder[item] for i, sublist in enumerate(input_y) if i not in skip_ind for item in sublist ], dtype=torch.long)
        #embeddings = embeddings.to(device)
        #y = y.to(device)
        return embeddings,y,len(label_encoder)





# ## defining tokenizer and bert model here
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
# model = AutoModel.from_pretrained('bert-base-uncased')
# model = model.to(device)




## TODO: for Lindsay: please fill in the following function so that
## we can derive the correct form

#wnl = WordNetLemmatizer()

def encoding_srl(srls:List[Dict], srl_ref=None):
    '''
    This function returns the encoded form of the SRL categories
    and an ordered dictionary for reference. The encoded SRL is a
    list where each entry contains a list of the SRL integer values 
    for the corresponding word. srl_ref is an ordered dictionary 
    whose keys correspond to SRL categories, which can be identified 
    by the integers corresponding to their positions/values. For 
    evaluation purposes we can provide a pre-defined srl_ref; all 
    SRL categories not seen during training should be categorized
    as 'unk'.
    '''
    train_ref = False
    if srl_ref is None: 
        srl_ref = {'unk':0}
        srl_ind = 1
        train_ref = True
    res = []
    for sent in srls:
        sent_res = [[] for i in range(len(sent[0]['frames']))]
        for i in range(len(sent)):
            vb = sent[i]['verb']
            #vb = wnl.lemmatize(vb,'v')
            for j in range(len(sent_res)):
                if sent[i]['frames'][j] != 'O':
                    fr = sent[i]['frames'][j][2:]
                    srl_key = vb + "_" + fr
                    if srl_key in srl_ref.keys():
                        n = srl_ref[srl_key]
                    elif train_ref:
                        srl_ref[srl_key] = srl_ind
                        n = srl_ind
                        srl_ind += 1
                    else:
                        n = 0
                    sent_res[j].append(n)
        res.append(sent_res)
    return srl_ref, res

# loading data from saved pickle files
# train_df = pd.read_pickle('../data/pmb_silver/silver_train.pkl')
# labels = np.concatenate(train_df['ccg_tags'].tolist())
# print(len(np.unique(labels)))
# train_input = train_df['text'].tolist()

# seq_lens = [len(item) for item in train_input]
# ## after doing some digging I decide to set max_len to be 75 to save computation
# ## This is because the total instances = 66582 and filtered instances = 66364
# max_len = 32
# # filtered = [i for i in seq_lens if i <=75]
# # print(len(seq_lens))
# # print(len(filtered))

# train_ccg = train_df['ccg_tags'].tolist()

# train_st = train_df['semantics_tags'].tolist()

# ## input for the model
# tokens, train_seq,train_mask = bert_tokenization(train_input, tokenizer,max_len=32)
# # print(train_seq,train_mask)
# # print(train_input[30])
# # detok = tokenizer.decode(train_seq[0])
# # print(detok)
# # sys.exit()
# train_seq = train_seq.to(device)
# train_mask = train_mask.to(device)

# # freeze all the parameters
# for param in model.parameters():
#     param.requires_grad = False

# model.eval()

# with torch.no_grad():
#     outputs = model(train_seq,attention_mask=train_mask)


# train_output = outputs[0].detach().cpu()
# # print(train_output[0])
# # print(train_output[0].shape)
# # print(train_input[0])
# # print(train_mask[0])

# # sys.exit()

# torch.save(train_output,'../data/pmb_gold/gold_dev_embeddings.pt')



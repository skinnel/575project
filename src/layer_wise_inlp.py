from load_bert import DataProcessing


def __main__():
    print('running layer-wise inlp')
    gold_train = DataProcessing('gold', 'test')
    gold_train.bert_tokenize()
    train_emb = gold_train.get_layerwise_embeddings('load')
    print('layerwise embedding shape:')
    print(train_emb.shape)


if __name__ == '__main__':
    __main__()


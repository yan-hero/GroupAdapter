import re
import pickle
from torch.utils.data import TensorDataset,DataLoader,Dataset
import torch
import jieba
from transformers import BertTokenizer,RobertaTokenizer
from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict
from typing import Dict,List,Union


def read_pkl(path:str):
    with open(path,'rb') as f:
        result = pickle.load(f)
    return result

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()

def stop_words(stop_file = r'./utils/stop_words.txt'):

    with open(stop_file,'r',encoding='utf-8') as file:
        return [line.replace('\n','').strip() for line in file]


def clean_and_stopwords(dataset):
    stopwordslist = stop_words()
    for i,line in enumerate(dataset['text']):
        line = clean_str_sst(line)
        new_str = jieba.cut_for_search(line)
        new_str_list = []
        for word in new_str:
            if word not in stopwordslist:
                new_str_list.append(word)
        dataset['text'][i] = ' '.join(new_str_list)

    # #English do not use jieba
    # for i,line in enumerate(dataset['text']):
    #     line = clean_str_sst(line)
    #     new_str_list = []
    #     for word in line:
    #         if word not in stopwordslist:
    #             new_str_list.append(word)
    #     dataset['text'][i] = ' '.join(new_str_list)

    return dataset



def load_all_data(train, validate):
    vocab = defaultdict(int)
    all_text = list(train['text']) + list(validate['text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text

def add_unknown_words(word_vecs, vocab, min_df=1, k=200):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def get_W(word_vecs, k=200):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer.from_pretrained(vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != tokenizer.pad_token_id)
    return token_ids, masks

class Rumor_Data(Dataset):
    def __init__(self, dataset,Event_flag=False):
        self.text = dataset['text']
        self.image = dataset['img_vec']
        self.mask = dataset['mask']
        self.label = torch.tensor(dataset['label'],dtype=torch.int64)
        self.flag = Event_flag
        if self.flag:
            self.event_label = torch.tensor(dataset['event_label'],dtype=torch.int64)

        if self.flag:
            print('TEXT: %d, Image: %d, Label: %d,Event_label: %d'
                  % (len(self.text), len(self.image), len(self.label),len(self.event_label)))
        else:
            print('TEXT: %d, Image: %d, Label: %d'
               % (len(self.text), len(self.image), len(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.flag:
            return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx],self.event_label[idx]
        else:
            return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx]



class Bert_data():
    def __init__(self,max_len, vocab_file,batch_size,num_workers):
        self.max_len = max_len
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.num_workers = num_workers

    def load_data(self,datasets:Dict[str,List[Union[str,torch.Tensor]]],shuffle = False,Event_flag=False):
        # img_vec = torch.stack(datasets['img_vec'],dim=0)
        # label = torch.as_tensor(datasets['label'])

        content = datasets['text']
        emb_ids, sent_masks = word2input(content, self.vocab_file, self.max_len)
        datasets['text'] = emb_ids
        datasets['mask'] = sent_masks
        dataset = Rumor_Data(datasets,Event_flag)

        # dataset = TensorDataset(
        #             emb_ids,
        #             sent_masks,
        #             img_vec,
        #             label)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,

        )
        return dataloader

class W2V_data():
    def __init__(self,max_len, vocab_file:Dict[str,int],batch_size,num_workers):
        self.max_len = max_len
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.num_workers = num_workers

    def tokenization(self, content:list):
        tokens = []
        for c in content:
            tokens.append([w for w in c])
        return tokens

    def get_mask(self, tokens):
        masks = []
        for sent in tokens:
            if (len(sent) < self.max_len):
                masks.append([1] * len(sent) + [0] * (self.max_len - len(sent)))
            else:
                masks.append([1] * self.max_len)

        return torch.tensor(masks)

    def encode(self, tokens):
        embedding = []
        for sent in tokens:
            words_ids = []
            for word in sent[:self.max_len]:
                words_ids.append(self.vocab_file[word] if word in self.vocab_file else 0)
                # if word in self.vocab_file:
                #     words_ids.append(self.vocab_file[word])
                # else:
                #     words_ids.append(0)

            for i in range(len(words_ids), self.max_len):
                words_ids.append(0)
            embedding.append(words_ids)
        return torch.tensor(np.array(embedding, dtype=np.int32))

    def load_data(self,datasets:Dict[str,List[Union[str,torch.Tensor]]],shuffle = False,Event_flag=False):

        # img_vec = torch.stack(datasets['img_vec'],dim=0)
        # label = torch.as_tensor(datasets['label'])

        sent_tokens = self.tokenization(datasets['text'])
        sent_masks = self.get_mask(sent_tokens )
        emb_ids = self.encode(sent_tokens)

        datasets['text'] = emb_ids
        datasets['mask'] = sent_masks
        dataset =Rumor_Data(datasets,Event_flag)

        # dataset = TensorDataset(
        #             emb_ids,
        #             sent_masks,
        #             img_vec,
        #             label
        # )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

        return dataloader

def get_dataloader(args):
    train_data = read_pkl(args.train_data_path)
    valid_data = read_pkl(args.valid_data_path)

    #clean str and stopwords
    train_data = clean_and_stopwords(train_data)
    valid_data = clean_and_stopwords(valid_data)

    text_vocab,all_text = load_all_data(train_data,valid_data)
    # args.vocab_size = len(text_vocab)
    # args.max_len = len(max(all_text,key=len))
    # print('args.max_len',args.max_len)

    if args.emb_type == 'bert':
        emb_dim = args.bert_emb_dim
        dataloader = Bert_data(max_len=args.max_len, vocab_file=args.bert_vocab_file,
                    batch_size=args.batch_size,num_workers= args.num_workers)
    elif args.emb_type == 'w2v':
        emb_dim = args.w2v_emb_dim
        vocab_file = args.w2v_vocab_file
        w2v = pickle.load(open(vocab_file,'rb'),encoding='bytes')
        # w2v = read_pkl(vocab_file)`
        add_unknown_words(w2v,vocab=text_vocab,k=emb_dim)
        W, word_idx_map = get_W(w2v,k=emb_dim)
        args.vocab_size = len(W)
        dataloader = W2V_data(max_len=args.max_len, vocab_file=word_idx_map,batch_size=args.batch_size, num_workers= args.num_workers)

    print('model name: {};emb_type: {}; batchsize: {}; epoch: {};  emb_dim: {}'.format( \
        args.model_name,args.emb_type,args.batch_size,args.epochs,emb_dim))

    train_loader = dataloader.load_data(train_data,shuffle=True,Event_flag=args.Event_flag)
    valid_loader = dataloader.load_data(valid_data,shuffle=False,Event_flag=args.Event_flag)

    if args.emb_type == 'bert':
        return train_loader,valid_loader
    elif args.emb_type == 'w2v':
        return train_loader,valid_loader,W


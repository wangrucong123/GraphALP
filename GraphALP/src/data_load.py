import json
import torch
import random
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import utils

# Load Cora Dataset
def load_cora(basepath,num_per_class=20, num_im_class=3, im_ratio=0.5,im_type="stpe"):

    print('Loading cora dataset...')
    
    idx_features_labels = np.genfromtxt(f"{basepath}/data/Cora/cora.content", dtype=np.dtype(str))
    edges_unordered = np.genfromtxt(f"{basepath}/data/Cora/cora.cites", dtype=np.int32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    classes_dict = {'Neural_Networks': 2, 'Reinforcement_Learning': 4, 'Probabilistic_Methods': 3, 'Case_Based': 0, 'Theory': 6, 'Rule_Learning': 5, 'Genetic_Algorithms': 1}

    labels = np.array(list(map(classes_dict.get, labels)))

    idx_dict = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = torch.FloatTensor(np.array(normalize(features).todense()))
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(np.array(adj.todense()))
    if im_type == "step":
        train_idx, val_idx, test_idx = get_step_data_cora(labels, num_per_class, num_im_class, im_ratio)
    elif im_type == "natural":
        pass
    else:
        raise ValueError(
            f"imb_type must be one of ['step', 'natural'], got {im_type}."
        )

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)


    return train_idx, val_idx, test_idx, adj, features, labels,edges


def load_citeseer(basepath,num_per_class=20, num_im_class=3, im_ratio=0.5,im_type="stpe"):
   

    print('Loading citeseer dataset...')
    idx_features_labels = np.genfromtxt(f"{basepath}/data/citeseer/citeseer.content", dtype=np.dtype(str))
    edges_unordered = np.genfromtxt(f"{basepath}/data/citeseer/citeseer.cites", dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    set_labels = set(labels)
    classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}
    
    classes_dict = {'Agents': 1, 'AI': 0, 'DB': 2, 'IR': 4, 'ML': 5, 'HCI': 3}
    labels = np.array(list(map(classes_dict.get, labels)))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.dtype(str)).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = torch.FloatTensor(np.array(adj.todense()))
    edges = torch.nonzero(adj, as_tuple=False)
    edges = edges[edges[:, 0] < edges[:, 1]].numpy()

    features = normalize(features)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)


    if im_type == "step":
        train_idx, val_idx, test_idx = get_step_data_cora(labels, num_per_class, num_im_class, im_ratio)
    elif im_type == "natural":
        pass
    else:
        raise ValueError(
            f"imb_type must be one of ['step', 'natural'], got {im_type}."
        )

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, adj, features, labels,edges



# Load Wiki-CS Dataset
def load_wiki_cs(basepath,num_per_class=20, num_im_class=3, im_ratio=0.5,im_type="stpe"):
    raw = json.load(open(f'{basepath}/data/wiki-cs/data.json'))
    features = torch.FloatTensor(np.array(raw['features']))
    labels = torch.LongTensor(np.array(raw['labels']))

    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i, nbs in enumerate(raw['links'])]))
    src, dst = tuple(zip(*edge_list))
    adj = np.unique(np.array([src, dst]).T, axis=0)
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = torch.FloatTensor(np.array(adj.todense()))
    edges = torch.nonzero(adj, as_tuple=False)
    edges = edges[edges[:, 0] < edges[:, 1]].numpy()
    
    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)

        c_num = len(c_idx)
        if c_num < 4:
            if c_num < 3:
                print("too small class type")
            batch_train = 1
            batch_val = 1
            batch_test = 1
        else:
            batch_train = int(c_num/4)
            batch_val = int(c_num/4)
            batch_test = int(c_num/4)

        train_idx = train_idx + c_idx[:batch_train]
        val_idx = val_idx + c_idx[batch_train:batch_train+batch_val]
        test_idx = test_idx + c_idx[batch_train+batch_val:batch_train+batch_val+batch_test]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, adj, features, labels,edges

def load_pubmed(basepath,num_per_class=20, num_im_class=3, im_ratio=0.5,im_type="stpe"):
    print('Loading pubmed dataset...')
    
    idx_features_labels = np.genfromtxt(f"{basepath}/data/pubmed/pubmed.content", dtype=np.dtype(str))
    edges_unordered = np.genfromtxt(f"{basepath}/data/pubmed/pubmed.cites", dtype=np.int32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]   
    


    classes_dict = {'2':0,'1':1,'3':2}   
    
    labels = np.array(list(map(classes_dict.get, labels)))
    
    idx_dict = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


    

    features = torch.FloatTensor(np.array(normalize(features).todense()))

    labels = labels.astype(float)
    labels = torch.LongTensor(labels)
    
    num_per_class_list = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - num_im_class:
            num_per_class_list.append(int(num_per_class * im_ratio))
        else:
            num_per_class_list.append(num_per_class)


    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []
    train_nodes = []
    attrm_idx = []
    
    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))

        train_idx = train_idx + c_idx[:num_per_class_list[i]]
        val_idx = val_idx + c_idx[num_per_class_list[i]:num_per_class_list[i] + 25]
        test_idx = test_idx + c_idx[num_per_class_list[i] + 25:num_per_class_list[i] + 80]
        train_nodes.append(c_idx[:num_per_class_list[i]])
        attrm_idx = attrm_idx + c_idx[num_per_class_list[i]:]
        

    random.shuffle(train_idx)
    adj = torch.FloatTensor(np.array(adj.todense()))
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    attrm_idx = torch.LongTensor(attrm_idx)

    return train_idx, val_idx, test_idx, adj, features, labels,edges





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_step_data_cora(labels,num_per_class, num_im_class, im_ratio):
    num_per_class_list = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - num_im_class:
            num_per_class_list.append(int(num_per_class * im_ratio))
        else:
            num_per_class_list.append(num_per_class)

    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)

        train_idx = train_idx + c_idx[:num_per_class_list[i]]
        val_idx = val_idx + c_idx[num_per_class_list[i]:num_per_class_list[i] + 25]
        test_idx = test_idx + c_idx[num_per_class_list[i] + 25:num_per_class_list[i] + 80]

    return train_idx,val_idx,test_idx

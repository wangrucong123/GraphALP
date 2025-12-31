import argparse
import os
import random
import numpy as np
from copy import deepcopy
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score, f1_score,balanced_accuracy_score,recall_score
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import LLmUp, models
from models import  Edge_Predictor, MLP,MLP1,CMLP#,MLP_E;
import torch.optim as optim

def evaluation(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    accuracy = correct.item() * 1.0 / len(labels)

    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1).detach().cpu().numpy(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu().numpy(), F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy(), average='macro')

    pred = logits.argmax(dim=1)
    y_pred = pred.cpu().numpy()
    y_true = labels.cpu().numpy()
    recalls = recall_score(y_true, y_pred, average=None)
    gmean = np.prod(recalls) ** (1.0 / len(recalls))

    macro_F = f1_score(labels.detach().cpu().numpy(), torch.argmax(logits, dim=-1).detach().cpu().numpy(), average='macro')
    
    return accuracy, gmean, macro_F


def adj_mse_loss(adj_rec, adj_tgt, param):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2
    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    if param['dataset'] == 'cora':
        return loss * 1e-3
    else:
        return loss / adj_tgt.shape[0]



def get_cate(cata,dataset):
    if dataset == "cora":
        coraCate = ["Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
                    "Reinforcement_Learning", "Rule_Learning", "Theory"]
        return coraCate[cata]
    elif dataset == "citeseer":
        citeseerCate = ["Agents", "Artificial Intelligence", "Database", "Human-Computer Interaction", "Machine Learning",
                    "Information Retrieval"]
        return citeseerCate[cata]
    elif dataset == "wiki-cs":
        wikiCate = ["Artificial Intelligence","Computer Vision","Cybersecurity","Data Science","Databases","Hardware & Architecture","	Human-Computer Interaction","	Machine Learning","Natural Language Processing","Theoretical Computer Science"]

        return wikiCate[cata]



def LLMs_upsample(cluEm,labels,idx_train,edges,param,modelLLm,tokenizer,device,basepath):
    """
        Perform  upsampling using LLM-generated embeddings.

        Args:
            cluEm (Tensor):  embeddings.
            labels (Tensor): Node labels.
            idx_train (Tensor): Training node indices.
            edges (ndarray): Existing graph edges.
            param (dict): Hyperparameter dictionary.
            modelLLm: Pretrained LM embedding model.
            tokenizer: Tokenizer for LM.
            device (torch.device): CUDA or CPU.
            basepath (str): Path to model directory.

        Returns:
            labels (Tensor): Updated labels with upsampled nodes.
            idx_train (Tensor): Updated training indices.
            newEm (Tensor): Concatenated original and synthetic embeddings.
            predict_edge_index (Tensor): Index of predicted new edges between synthetic and original nodes.
        """
    newEm = cluEm.detach()
    class_num_list = []
    cndic2 = {}
    temp = None

    # Count instances per class
    for K in range(param["classNum"]):
        cndic2[K] = labels[idx_train][labels[idx_train] == K].shape[0]
        class_num_list.append(labels[idx_train][labels[idx_train] == K].shape[0])
    # Determine target max class count
    maxClass = int(max(class_num_list)*param["upsc"])
    # Upsample each minority class
    for k in range(len(class_num_list)):
        num = maxClass - class_num_list[k]
        if num > 0:
            emLLm ,num= LLmUp.llm_upsampling(device,basepath,param["dataset"], cate=k, num=num, model=modelLLm, tokenizer=tokenizer)
            emLLm  = emLLm.to(device)
            # Add new indices to training set and labels
            idx_new = idx_train.new(np.arange(labels.shape[0], labels.shape[0] + num))
            idx_train = torch.cat((idx_train, idx_new), 0)
            new_labels = labels.new(torch.Size((num, 1))).reshape(-1).fill_(k).to(device)
            labels = torch.cat((labels, new_labels), dim=0)
            # Accumulate new LLM-generated embeddings
            if temp is None:
                temp = emLLm
            else:
                temp = torch.cat((temp, emLLm), dim=0)
    try:
        if temp is not None:

            # Align synthetic embeddings Z1 with original Z2
            mlp_z1 = MLP1(temp.shape[1], 256).to(device)  
            mlp_z2 = MLP1(newEm.shape[1], 256).to(device)  
            mlpC = CMLP(256,param['nhid']).to(device)  

            h1 = mlp_z1(temp.to(device))
            h2 = mlp_z2(newEm.to(device))

            h1 = mlpC(h1.to(device))
            h2 = mlpC(h2.to(device))

            h1 = F.normalize(h1, p=2, dim=-1)
            h2 = F.normalize(h2, p=2, dim=-1)
            # Find similar pairs above threshold
            similarity_matrix = F.cosine_similarity(h1.detach().unsqueeze(1), h2.detach().unsqueeze(0), dim=-1)
            new_edge = torch.nonzero(similarity_matrix > param['sim_ratio']).cpu()
            new_edge[:, 0] = new_edge[:, 0] + h2.shape[0]
            predict_edge_index = new_edge.transpose(0, 1)
            
            return labels.to(device),idx_train.to(device),torch.cat([h2.detach(), h1.detach()], dim=0).to(device),predict_edge_index.to(device)
        else:
            return labels.to(device),idx_train.to(device),cluEm.to(device),None
    finally:

        torch.cuda.empty_cache()
        pass

import torch.nn as nn


def Edge(newEm,adj,predict_edge_index,device,edges):
    """
        Train an edge predictor and update the graph adjacency with predicted edges.
        Args:
            newEm (Tensor): Node embeddings (including new LLM-generated ones).
            adj (Tensor): Original adjacency matrix.
            predict_edge_index (Tensor): Candidate edges to evaluate.
            device (torch.device): CUDA or CPU.
            edges (Tensor): Original edge index (transpose(0,1)).

        Returns:
            adj (Tensor): Updated adjacency matrix.
            input_edge_index (Tensor): Combined edge indices (original + predicted).
        """
    try:
        update_edge = edges.transpose(0, 1).to(device)
        E = update_edge.size(1)
        # Sample equal number of non-edges (negatives) as negatives
        missing_edge = torch.nonzero(adj != 1)
        missing_edge_idx = np.arange(0, missing_edge.size(0))
        np.random.shuffle(missing_edge_idx)
        missing_edge_index = missing_edge[missing_edge_idx[0:E]].transpose(0, 1).to(device)
        # Label positive and negative edges
        edge_y = torch.cat([torch.ones(E), torch.zeros(E)])
        train_edge = torch.cat([update_edge.cpu(), missing_edge_index.cpu()], dim=1).to(device)

        newEm = newEm.to(device)

        # Train a binary edge predictor using BCE loss
        hidden_dim = 32
        Edge = Edge_Predictor(newEm.shape[1], hidden_dim, predict_edge_index).to(device)
        optimizer_e = torch.optim.Adam(Edge.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.BCELoss()
        
        for epoch in range(1, 300):
            Edge.train()
            optimizer_e.zero_grad()
            out_edge = Edge(newEm, train_edge)
            loss_edge = criterion(out_edge.to(device), edge_y.to(device).unsqueeze(1))
            loss_edge.backward()
            optimizer_e.step()
        # Get new edges with high confidence (using .predict_bce method)
        new_edge = Edge.predict_bce()
        input_edge_index = torch.cat([edges.transpose(0, 1).to(device), new_edge], dim=1)
        input_edge_index = torch.cat([input_edge_index, new_edge[[1, 0]]], dim=1)
        # Convert to dense adjacency matrix
        sparse_matrix = torch.sparse_coo_tensor(input_edge_index.to(torch.long).to(device),torch.ones(input_edge_index.size(1)).to(device),torch.Size([newEm.shape[0], newEm.shape[0]]))
        adj = sparse_matrix.to_dense()
        
        return adj,input_edge_index
    finally:
            torch.cuda.empty_cache()
            pass


def getModelAndEncode(param,features,labels,args):
    if param['setting'] != 'embed_smote':
        if param['model'] == 'sage':
            encoder = models.Sage_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'])
            classifier = models.Sage_Classifier(nembed=param['nhid'], nhid=param['nhid'],
                                                nclass=labels.max().item() + 1, dropout=param['dropout'])
            
            #models.Sage_Classifier
        elif param['model'] == 'gcn':
            encoder = models.GCN_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                    dropout=param['dropout'],num_layers = param['num_layers'])
            classifier = models.GCN_Classifier(nembed=param['nhid'], nhid=param['nhid'], nclass=labels.max().item() + 1,
                                               dropout=param['dropout'])
        elif args.model == 'sem':
            encoder = models.SEM_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                    dropout=param['dropout'], nheads=param['nhead'], graph_mode=param['graph_mode'])
            classifier = models.SEM_Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                               dropout=param['dropout'])
        elif args.model == 'gat':
            encoder = models.GAT_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                    dropout=param['dropout'], nheads=param['nhead'])
            classifier = models.GAT_Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                               dropout=param['dropout'], nheads=param['nhead'])
    else:
        if args.model == 'sage':
            encoder = models.Sage_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                      dropout=param['dropout'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])
        elif args.model == 'gcn':
            encoder = models.GCN_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])
        elif args.model == 'sem':
            encoder = models.SEM_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'], nheads=param['nhead'], graph_mode=param['graph_mode'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])
        elif args.model == 'gat':
            encoder = models.GAT_En2(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                     dropout=param['dropout'], nheads=param['nhead'])
            classifier = models.Classifier(nembed=args.nhid, nhid=param['nhid'], nclass=labels.max().item() + 1,
                                           dropout=param['dropout'])
    return encoder,classifier




def getArgs():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--lk', type=bool, default=False)
    
    parser.add_argument('--num_im_class', type=int, default=3, choices=[3, 14, 10])

    parser.add_argument('--model', type=str, default='sage', choices=['sage', 'gcn', 'gat'])
    parser.add_argument('--setting', type=str, default='pre-train',
                        choices=[ 'pre-train', 'fine-tune'])

    parser.add_argument('--mode', type=str, default='continuous_edge', choices=['discrete_edge', 'continuous_edge'])
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--graph_mode', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2025)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--im_type', type=str, default="step",choices=['step','natural'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer', 'wiki-cs','pubmed'])
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--noise_type', type=str,
                        default='uniform',
                        choices=['clean', 'uniform', 'pair', 'random'], help='Type of label noise')

    parser.add_argument('--noise_rate', type=float,
                        default=0.3,
                        help='Label noise rate')
    parser.add_argument('--im_ratio', type=float, default=0.7)
    parser.add_argument('--isllm', action='store_true', default=True)
    parser.add_argument('--isp', action='store_true', default=True)
    parser.add_argument('--re', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=50, help='early stop.')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--pep', type=int, default=10)
    
    parser.add_argument('--P_sel_train', type=float, default=0.87)
    parser.add_argument('--la', type=float, default=1)
    parser.add_argument('--lx1', type=float, default=1)
    parser.add_argument('--lx2', type=float, default=1)
    parser.add_argument('--sim_ratio', type=float, default=0.8)
    parser.add_argument('--upsc', type=float, default=0.7)

    args, _ = parser.parse_known_args()
    return args



# def getArgs():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--load', type=str, default=None)
#     parser.add_argument('--lk', type=bool, default=False)
    

#     parser.add_argument('--num_im_class', type=int, default=3, choices=[3, 14, 10])

#     parser.add_argument('--model', type=str, default='sage', choices=['sage', 'gcn', 'gat'])
#     parser.add_argument('--setting', type=str, default='pre-train',
#                         choices=['pre-train', 'fine-tune'])

#     parser.add_argument('--mode', type=str, default='continuous_edge', choices=['discrete_edge', 'continuous_edge'])
#     parser.add_argument('--nhead', type=int, default=4)

   
#     parser.add_argument('--epochs', type=int, default=2025)
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--weight_decay', type=float, default=5e-4)
    
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--im_type', type=str, default="step",choices=['step','natural'])
#     parser.add_argument('--num_layers', type=int, default=2)
#     parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora','citeseer', 'wiki-cs','pubmed','amap','Coauthor_CS'])
#     parser.add_argument('--dropout', type=float, default=0.1)
    
#     parser.add_argument('--nhid', type=int, default=128)
#     parser.add_argument('--noise_type', type=str,
#                         default='uniform',
#                         choices=['clean', 'uniform', 'pair', 'random'], help='Type of label noise')

#     parser.add_argument('--noise_rate', type=float,
#                         default=0.3,
#                         help='Label noise rate')
#     parser.add_argument('--im_ratio', type=float, default=0.7)
#     parser.add_argument('--isllm', action='store_true', default=True)
#     parser.add_argument('--isp', action='store_true', default=True)
#     parser.add_argument('--re', action='store_true', default=True)
#     parser.add_argument('--patience', type=int, default=50, help='early stop.')
#     parser.add_argument('--cuda', type=int, default=0)
#     parser.add_argument('--pep', type=int, default=25)
#     parser.add_argument('--P_sel_train', type=float, default=0.83)  # 0.97
#     parser.add_argument('--la', type=float, default=1)
#     parser.add_argument('--lx1', type=float, default=1)
#     parser.add_argument('--lx2', type=float, default=1)
#     parser.add_argument('--sim_ratio', type=float, default=0.99)
#     parser.add_argument('--upsc', type=float, default=0.8)# cora 0.8
#     args, _ = parser.parse_known_args()
#     return args

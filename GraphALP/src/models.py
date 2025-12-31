import os
import math
import pymetis
import collections
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class SageConv(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.linear = nn.Linear(in_features*2, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)

    def forward(self, features, adj):
        neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        features = torch.cat([features, neigh_feature], dim=-1)
        output = self.linear(features)

        return output





    def compute_semantic_loss(self):

        labels = [torch.ones(1)*i for i in range(self.nheads)]
        labels = torch.cat(tuple(labels), 0).long().to(device)

        factors_feature = torch.cat(tuple(self.out_descriptor), 0)
        pred = self.classifier(factors_feature)

        if self.graph_mode == 0:
            pred = nn.Softmax(dim=1)(pred)
            loss_sem = self.loss_fn(pred, labels)
        elif self.graph_mode == 1:
            loss_sem_list = []
            for i in range(self.nheads-1):
                for j in range(i+1, self.nheads):
                    loss_sem_list.append(torch.cosine_similarity(pred[i], pred[j], dim=0))
            loss_sem_list = torch.stack(loss_sem_list)
            loss_sem = torch.mean(loss_sem_list)
        else:
            loss_sem_list = []
            for i in range(self.nheads-1):
                for j in range(i+1, self.nheads):
                    loss_sem_list.append(self.loss_fn(pred[i], pred[j]))
            loss_sem_list = torch.stack(loss_sem_list)
            loss_sem = - torch.mean(loss_sem_list)    

        return loss_sem
                




class GCN_En(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout,num_layers=2):
        super(GCN_En, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        #self.gc2 = GraphConvolution(nhid, nembed)
        self.gc2 = GraphConvolution(nhid, nembed) if num_layers == 2 else None
        self.dropout = dropout
        self.num_layers = num_layers


    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        if self.num_layers == 2:
                x1 = F.relu(self.gc2(x1, adj))
                x1 = F.dropout(x1, self.dropout, training=self.training)

        return x1


class GCN_En2(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_Classifier(nn.Module):
    
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nhid, nembed)
        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()



    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std = 0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.mlp(x)

        return x1


class Sage_En(nn.Module):

    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()
        
        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nembed)
        self.dropout = dropout

    
    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x
        

class Sage_En2(nn.Module):
    
    def __init__(self, nfeat, nhid, nembed, dropout):
        
        super(Sage_En2, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nembed)
        self.dropout = dropout
    
    

    def forward(self, x, adj):
        
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x




class Sage_Classifier(nn.Module):

    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nhid, nembed)
        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std = 0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x







class GAT_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, nheads=4):
        super(GAT_En, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, int(nhid / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear = nn.Linear(nhid, nembed)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.05)

    def forward(self, x, adj):

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.linear(x))

        return x


class GAT_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, nheads=4):
        super(GAT_En2, self).__init__()

        self.attentions_1 = [GraphAttentionLayer(nfeat, int(nhid / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention1_{}'.format(i), attention)

        self.linear_1 = nn.Linear(nhid, nembed)
        self.dropout = dropout

        self.attentions_2 = [GraphAttentionLayer(nembed, int(nembed / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention2_{}'.format(i), attention)

        self.linear_2 = nn.Linear(nembed, nembed)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_1.weight, std=0.05)
        nn.init.normal_(self.linear_2.weight, std=0.05)


    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions_1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.linear_1(x))

        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.linear_2(x))

        return x


class GAT_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, nheads=4):
        super(GAT_Classifier, self).__init__()

        self.attentions = [GraphAttentionLayer(nembed, int(nhid / nheads), dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear = nn.Linear(nhid, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=0.05)
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.elu(self.linear(x))
        x = self.mlp(x)

        return x


class Decoder(nn.Module):

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out






class Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp = nn.Linear(nembed, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = self.mlp(x)

        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,bias=False):
        super(MLP, self).__init__()
        # cora 256
        self.fc1 = nn.Linear(input_dim, 256)  
        self.fc2 = nn.Linear(256, output_dim) 
 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim,bias=False):
        super(MLP1, self).__init__()
        # cora 256
        self.fc1 = nn.Linear(input_dim, output_dim)  
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

class CMLP(nn.Module):
    def __init__(self, input_dim, output_dim,bias=False):
        super(CMLP, self).__init__()
        
        self.fc2 = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x





class Edge_Predictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, predict_edge_index):
        super(Edge_Predictor, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.predict_edge_index = predict_edge_index
        self.threshold = nn.Parameter(torch.tensor(0.5))  

    def forward(self, x, train_edge_idx):
        self.x = x
        train_x = self.x[train_edge_idx[0]]
        train_y = self.x[train_edge_idx[1]]
        train_edge = self.MLP(torch.cat([train_x, train_y], dim=1))
        return train_edge

    def predict_bce(self):
        predict_x = self.x[self.predict_edge_index[0]]
        predict_y = self.x[self.predict_edge_index[1]]
        
        predict_new_edge_kc2n = self.MLP(torch.cat([predict_x, predict_y], dim=1)).squeeze(1)
        predict_new_edge_n2kc = self.MLP(torch.cat([predict_y, predict_x], dim=1)).squeeze(1)
        predict_new_edge = predict_new_edge_kc2n + predict_new_edge_n2kc
       

        mean_value = predict_new_edge.mean()  # 计算均值
        k = (predict_new_edge > mean_value).sum().item()
        
        values, indices = torch.topk(predict_new_edge, k=k)
        predict_edge = self.predict_edge_index[:, indices]

        return predict_edge.detach()


class GCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim,  out_channels)

    def forward(self, x, edge_index, drop):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=drop, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x ,dim=1)
        return F.log_softmax(x, dim=1)



class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


#============================================= DAO
#gcn_encode
class MLP_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(MLP_En, self).__init__()
        self.norm_input = Norm(nfeat)
        self.mlp1=MLP(nfeat, nhid)
        self.MLPfirst = nn.Linear(nfeat, nhid)

        self.MLPlast = nn.Linear(nhid, 7)

    def forward(self, x):

        x1 = F.relu(self.mlp1(x))
        x_input = self.norm_input(x)
        x2 = self.MLPfirst(x_input)
        x2 = F.dropout(x2, 0.4, training=True) # cora 0.4 
        
        CONN_INDEX = F.relu(self.MLPlast(x2))

        return x1,F.softmax(CONN_INDEX, dim=1)


class Decoder_MLP(nn.Module):

    def __init__(self, nembed, outFeat, dropout=0.1,dataset = "cora"):
        super(Decoder_MLP, self).__init__()

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()
        if dataset =="cora" or dataset == "citeseer" :
             #########cora citeseer  cora 128 
            self.fc4 = nn.Linear(nembed, 512)
            self.fc5 = nn.Linear(512, outFeat)
        elif dataset == "citeseer":
            self.fc4 = nn.Linear(nembed, 512)
            self.fc5 = nn.Linear(512, outFeat)
        elif dataset == "BlogCatalog":
            #################blogcatalog
            self.fc4 = nn.Linear(nembed, 128)
            self.fc5 = nn.Linear(128, 64)
        elif dataset == "wiki-cs":
            #########wiki-cs
            self.fc4 = nn.Linear(nembed, 512)
            self.fc5 = nn.Linear(512, outFeat)
        elif dataset == "pubmed":
            self.fc4 = nn.Linear(nembed, 512)
            self.fc5 = nn.Linear(512, outFeat)
        else:
            self.fc4 = nn.Linear(nembed, 128)
            self.fc5 = nn.Linear(128, outFeat)


    def decode(self, z):
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))
        x_reconst = self.decode(node_embed)

        return adj_out, x_reconst
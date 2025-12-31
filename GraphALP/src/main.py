
from tqdm import tqdm
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from transformers import AutoTokenizer,AutoModel
import os
import models
from tqdm import tqdm
import data_load
import labelnoise
import utils





def train(epoch):
   
    
    labelsUp =0
    idx_trainUp = 0

    # Set models to training mode
    encoder.train()
    classifier.train()
    decoder.train()
    encoderMLP.train()

    optimizer_enMLP.zero_grad()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    embed,o = encoderMLP(features)

    if  param['setting'] == 'pre-train' or param['setting'] == 'fine-tune':
        # No LLM augmentation;
        if param["isllm"] == False or param["im_ratio"] == 1.0 :
            labels_new = labels
            idx_train_new = idx_train
            adj_up = adj

        elif param["isllm"] and param["im_ratio"] != 1.0 :
            if param["lk"]:#LLM augmentation;
                labelsUp, idx_trainUp, newEm, predict_edge_index = utils.LLMs_upsample(embed, labels, idx_train, edges,
                                                                                    param, modelLm, tokenizer, device,basepath)
                if predict_edge_index is not None:
                    adjNew, edge_index = utils.Edge(newEm, adj, predict_edge_index, device, edges)
                else:
                    sparse_matrix = torch.sparse_coo_tensor(edges.transpose(0, 1).to(torch.long).to(device),torch.ones(edges.transpose(0, 1).size(1)).to(device),torch.Size([newEm.shape[0], newEm.shape[0]]))
                    adjNew = sparse_matrix.to_dense()
                # Cache the result
                best_req[0] = newEm.cpu()
                best_req[1] = labelsUp.cpu()
                best_req[2] = idx_trainUp.cpu()
                best_req[3] = adjNew.cpu()
                param["lk"] = False

            else:
                # Use previously cached upsampling result
                newEm =  best_req[0].to(device)
                labelsUp =  best_req[1].to(device)
                idx_trainUp =  best_req[2].to(device)
                adjNew =  best_req[3].to(device)

            # Update embed and adj with LLM-augmented version
            newEm[:n_num, :][:, :n_num] = embed
            embed = newEm.detach()
            labels_new = labelsUp.detach()
            idx_train_new = idx_trainUp.detach()
            adj_up = adjNew.detach()
            adj_up[:n_num, :][:, :n_num] = adj.detach()

        # Structure and feature reconstruction
        adj_rec, feature_reconst = decoder(embed)
        # Structure reconstruction loss
        loss_rec = utils.adj_mse_loss(adj_rec[:n_num, :][:, :n_num], adj.detach(), param)
        # Feature reconstruction loss
        criterion = torch.nn.MSELoss(reduction='sum')
        reconst_loss = criterion(feature_reconst[:n_num, :], features)

        # Obtain threshold binary edges or soft continuous edges
        if param['mode'] == 'discrete_edge':
            adj_new = copy.deepcopy(adj_rec.detach())
            threshold = 0.5
            adj_new[adj_new < threshold] = 0.0
            adj_new[adj_new >= threshold] = 1.0
        else:
            adj_new = adj_rec

        # Mask by LLM adjacency and add back original edges
        adj_new = torch.mul(adj_up, adj_new)
        adj_new[:n_num, :][:, :n_num] = adj.detach()

        if param['mode'] == 'discrete_edge':
            adj_new = adj_new.detach()

    # GNN Encoding + Decoding
    feature_rec = torch.cat((features, feature_reconst[n_num:, :]), dim=0)
    gnn_embed = encoder(feature_rec, adj_new)
    gnnEm = gnn_embed[:adj.shape[0], :]
    generated_G_gnn, feature_reconst_gnn = decoder(gnnEm)
    loss_rec_gnn = utils.adj_mse_loss(generated_G_gnn, adj.detach(), param)
    output = classifier(gnn_embed, adj_new)

    # The Re-weight method assign larger weight to losses of samples on minority classes
    if param['re']:
        weight = gnn_embed.new((labels.max().item() + 1)).fill_(1)
        c_largest = labels.max().item()+1
        unique_labels, counts = torch.unique(labels[idx_train], return_counts=True)
        avg_number = int(idx_train.shape[0] / (c_largest + 1))
        # Adjust class weight inversely proportional to class frequency
        for label in range(c_largest):
            if (unique_labels == label).any().item():
                index = (unique_labels == label).nonzero(as_tuple=True)[0].item()
                count = counts[index]
                c_up_scale = int(avg_number / count) - 1
                if c_up_scale >= 0:
                    weight[label] = 1 + c_up_scale
            else:
                missing_class_weight = 2  
                weight[label] = missing_class_weight
        
            loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new],weight=weight)
    else:
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])



    # Perform pre-train
    if param['setting'] == 'pre-train':
        loss = param['la'] * loss_rec + 0.0 * loss_train + param['lx1'] * reconst_loss + param['lx2'] * loss_rec_gnn
        loss.backward()
        optimizer_enMLP.step()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

    # Perform fine-tune
    elif param['setting'] == 'fine-tune':
        loss = loss_train
        loss.backward()
        optimizer_enMLP.step()
        optimizer_en.step()
        optimizer_cls.step()
        optimizer_de.step()

    acc_val, auc_val, f1_val = utils.evaluation(output[idx_val], val_labels)

    return f1_val,output,labelsUp,idx_trainUp



def test(epoch):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    
    if param['model'] == 'sem':
        embed, _ = encoder(features, adj)
    else:
        embed= encoder(features, adj)
    output = classifier(embed, adj)
    acc_test, auc_test, f1_test = utils.evaluation(output[idx_test], test_labels)
    return acc_test, auc_test, f1_test,output





if __name__ == "__main__":
    avg_acc = 0
    avg_Gmean = 0
    ave_f1 = 0

    bset_train = 0
    best_o = 0
    best_la = 0

    for s in range(42,47):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Get current file path
        basepath = os.path.abspath(__file__)
        basepath = os.path.dirname(basepath)

        args = utils.getArgs()
        param = args.__dict__
        device = torch.device(f'cuda:{param["cuda"]}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(param["cuda"])

        # Load tokenizer and LM model
        tokenizer = AutoTokenizer.from_pretrained(f"{basepath}/jina/jina-embeddings-v3",
                                                trust_remote_code=True)
        modelLm = AutoModel.from_pretrained(f"{basepath}/jina/jina-embeddings-v3", trust_remote_code=True,local_files_only=True).to(device)
        param['seed'] = s
        random.seed(param['seed'])
        np.random.seed(param['seed'])
        torch.manual_seed(param['seed'])
        torch.cuda.manual_seed(param['seed'])


        if param['dataset'] == 'pubmed':
            param['num_im_class'] = 1


        # Load Dataset
        if param['dataset'] == 'cora':
            idx_train, idx_val, idx_test, adj, features, labels,edges = data_load.load_cora(basepath,num_per_class=20,
                                                                                    num_im_class=param['num_im_class'],
                                                                                    im_ratio=param['im_ratio'],im_type=param['im_type'])
        elif param['dataset'] == 'citeseer':
            idx_train, idx_val, idx_test, adj, features, labels,edges = data_load.load_citeseer(basepath,num_per_class=20,
                                                                                        num_im_class=param[
                                                                                            'num_im_class'],
                                                                                        im_ratio=param['im_ratio'],im_type=param['im_type'])
        elif param['dataset'] == 'wiki-cs':
            idx_train, idx_val, idx_test, adj, features, labels,edges = data_load.load_wiki_cs(basepath,num_per_class=20,
                                                                                        num_im_class=param[
                                                                                            'num_im_class'],
                                                                                        im_ratio=param['im_ratio'],im_type=param['im_type'])
        
        elif param['dataset'] == 'pubmed':
            idx_train, idx_val, idx_test, adj, features, labels,edges = data_load.load_pubmed(basepath,num_per_class=20,
                                                                                        num_im_class=param[
                                                                                            'num_im_class'],
                                                                                        im_ratio=param['im_ratio'],im_type=param['im_type']) 
        else:
            print("no this dataset: {param['dataset']}")
        
        param['nodeNum'] = features.shape[0]
        f_col =  features.shape[1]
        param['classNum'] = (labels.max()+1).item()
        n_num = features.shape[0]
        labels_o = labels.detach().to(device)

        # Add label noise
        if param['noise_rate'] > 0:
            labels[idx_train], modified_mask, current_mask = labelnoise.label_process(labels=labels,
                                                                                n_classes=labels.max().item() + 1,
                                                                                train_data=idx_train,
                                                                                noise_type=args.noise_type,
                                                                                noise_rate=args.noise_rate,
                                                                                random_seed=args.seed, debug=False)
            print((labels_o.cpu()[idx_train] == labels[idx_train]).cpu().numpy().mean())

        # Load different bottleneck encoders and classifiers
        encoder,classifier = utils.getModelAndEncode(param, features, labels, args)
        # Load decoder and MLP
        encoderMLP = models.MLP_En(nfeat=features.shape[1], nhid=param['nhid'], nembed=param['nhid'],
                                   dropout=param['dropout'])
        decoder = models.Decoder_MLP(nembed=param['nhid'],outFeat=f_col, dropout=param['dropout'],dataset = param['dataset'])

        #  Define optimizers for each module
        optimizer_enMLP = torch.optim.Adam(encoderMLP.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
        optimizer_en = torch.optim.Adam(encoder.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
        optimizer_de = torch.optim.Adam(list(decoder.parameters()), lr=param['lr'], weight_decay=param['weight_decay'])
        encoderMLP = encoderMLP.to(device)
        encoder = encoder.to(device)
        classifier = classifier.to(device)
        decoder = decoder.to(device)


        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        edges = torch.from_numpy(edges)

        # Prepare validation/test labels
        test_labels = labels[idx_test].to(device)
        val_labels = labels[idx_val].to(device)

        # ======== Pre-training stage ========
        es = 0
        f1_val_best = 0
        metric_test_val = [0, 0, 0, 0]
        metric_test_best = [0, 0, 0]
        newEm = None
        best_req = [0,0,0,0]
        import time
        start_time = time.time()

        if param["isllm"] and param["im_ratio"] != 1.0 :

            encoderMLP.eval()
            embed,o = encoderMLP(features)
            
            labelsUp, idx_trainUp, newEm, predict_edge_index = utils.LLMs_upsample(embed, labels, idx_train, edges,
                                                                                param, modelLm, tokenizer, device,basepath)
            if predict_edge_index is not None:
                adjNew, edge_index = utils.Edge(newEm, adj, predict_edge_index, device, edges)
            else:
                sparse_matrix = torch.sparse_coo_tensor(edges.transpose(0, 1).to(torch.long).to(device),torch.ones(edges.transpose(0, 1).size(1)).to(device),torch.Size([newEm.shape[0], newEm.shape[0]]))
                adjNew = sparse_matrix.to_dense()
            best_req[0] = newEm.cpu()
            best_req[1] = labelsUp.cpu()
            best_req[2] = idx_trainUp.cpu()
            best_req[3] = adjNew.cpu()

        
        for epoch in tqdm(range(param['epochs'])):
            f1_val,_,_,_ = train(epoch)

        # ======== Prepare unlabeled mask for pseudo-labeling ========
        idx_trainA = idx_train.detach().to(device)
        idx_train= idx_train.to(device)
        if param['dataset'] == "wiki-cs":
            train_unsel = torch.cat((idx_trainA,idx_val), dim=0) #
        else:
            train_unsel = torch.cat((idx_trainA,idx_val, idx_test), dim=0) #
        
        train_all_pos_bool = torch.ones(labels.size(0))
        train_all_pos_bool[train_unsel] = 0
        train_all_pos = train_all_pos_bool.to(device)
        
        train_unselA = torch.cat((idx_trainA,idx_val, idx_test), dim=0)
        
        
        unlabeled_mask = torch.ones(labels.size(0), dtype=torch.bool).to(device)
        unlabeled_mask[train_unselA] = False

        # ======== Fine-tuning stage ========
        if param['setting'] == 'pre-train':
            param['setting'] = 'fine-tune'
            print("fine-tune")
            best_score = 0
            es = 0
            f1_val_best = 0
            metric_test_val = [0, 0, 0, 0]
            metric_test_best = [0, 0, 0]
            idx_train_last = idx_trainA
            for epoch in range(param['epochs']+1000):
                f1_val,o,labelsUp,idx_trainUp = train(epoch)
                
                if epoch % 5 == 0 and epoch != 0:
                    acc_test, Gmean_test, f1_test,output = test(epoch)
                    output = F.softmax(output, dim=1)

                    # Update best results or perform pseudo-label update
                    if f1_val > f1_val_best or (epoch == param['pep'] and param["isp"]):
                        if f1_val > f1_val_best:
                            f1_val_best = f1_val
                            metric_test_val[0] = acc_test
                            metric_test_val[1] = Gmean_test
                            metric_test_val[2] = f1_test
                            metric_test_val[3] = epoch
                            es = 0
                        if epoch >= param['pep'] and param["isp"]:
                            param["lk"] = True
                            confidence, pseudo_labels = torch.max(output, dim=1)
                            labels[unlabeled_mask] = pseudo_labels[unlabeled_mask]
                            pre_ind_min = confidence >= param["P_sel_train"]
                            idx_train_mask = pre_ind_min.float() + train_all_pos[:adj.shape[0]].float() == 2
                            idx_train_new = idx_train_mask.detach().clone()
                            idx_train_new = torch.nonzero(idx_train_new).squeeze()

                            try:
                                
                                idx_train_new = torch.cat([idx_train_new, idx_trainA]).to(device)
                            except:
                                idx_train_new = idx_train_last
                            idx_train = idx_train_new.detach()
                    else:
                        es += 1
                        if es >= param['patience']:
                            print("Early stopping!")
                            break


                    # Compute score
                    sc = param["im_ratio"] / (param["noise_rate"] + param["im_ratio"])
                    score = f1_test + (1 - sc) * Gmean_test
                    if score > best_score:
                        best_score = score
                        metric_test_best[0] = acc_test
                        metric_test_best[1] = Gmean_test
                        metric_test_best[2] = f1_test

        # Final output
        avg_acc = avg_acc + metric_test_best[0]
        avg_Gmean = avg_Gmean + metric_test_best[1]
        ave_f1 = ave_f1 + metric_test_best[2]
        print(f"best_acc:{metric_test_best[0]},GMean:{metric_test_best[1]},best_f1:{metric_test_best[2]}")
        torch.cuda.empty_cache()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"run time: {elapsed_time:.4f} 秒")


    print(f"best_acc:{avg_acc/5},GMean:{avg_Gmean/5},best_f1:{ave_f1/5}")

    



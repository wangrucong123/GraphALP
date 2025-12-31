# GraphALP：When Noisy Labels Meet Class Imbalance on Graphs: A Graph Augmentation Method with LLM and Pseudo Label

## Main Requirements
* python==3.9
* torch==3.0.1
* openai==1.78.1
* transformers==4.49.0
* numpy==1.24.3
* scipy==1.13.1
* ipdb==0.13.13
* ogb==1.3.6
* networkx==2.6
* pandas==2.2.3
* pymetis==2023.1.1
* torch-geometric==2.6.1

## LLM and LM
  * **LLM:**: DeepSeek-Chat
  * **LM**: jina-embedding-v3

## Dataset 
The experiments use four graph datasets: **Cora**、**CiteSeer**、**Wiki-CS**、**Pubmed**.

## Evaluation Metrics

- **Accuracy (ACC)**
- **Macro-F1** (for class balance)
- **Micro-F1** (for global correctness)
- **G-Mean** (Geometric mean of recall for all classes)

## How To Use
```bash
$ Install the required dependency packages
# Run training
$ python main.py --dataset cora --im_ratio 0.7 --noise_rate 0.3 --cuda 0 --sim_ratio 0.8  --upsc 0.7 --la 1  --lx1 1 --lx2 1
```




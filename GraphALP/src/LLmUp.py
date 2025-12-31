import os
import pickle
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI, embeddings
from sympy.physics.quantum.gate import normalized
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="Your token"
)



def Deepseek_get_categories(system_prompt,pt):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": pt
        }
    ]
    #gpt-4  deepseek-chat
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens= 1024 * 5,
        messages=messages,

    )

    classesList = response.choices[0].message.content.split("|")
    return classesList


def text_tor(device,text,model,tokenizer):
    """
        Tokenize and embed a list of input texts using a transformer model.
    """
    inputs = tokenizer(text,return_tensors="pt",padding=True,truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs


def llm_upsampling(device,basepath,dataset,cate,num,model=None,tokenizer=None):
    """
       Load and embed text attributes (category + title + abstract) generated for class-balanced upsampling using LLM.
       Args:
           device (torch.device): Device to load embeddings onto.
           basepath (str): Project root path.
           dataset (str): Dataset name (e.g., "cora").
           cate (int): Target category index.
           num (int): Number of samples to generate or retrieve.
           model: Embedding model (e.g., jina-embedding-v3).
           tokenizer: Corresponding tokenizer for the model.

       Returns:
           - Embedding  for LLM-generated texts.
           - Actual number of valid generated samples.
       """
    if dataset == "cora":
        dataset = "Cora"
        system_prompt = """
                    #### Positioning
                   - Smart Assistant Name: Citation Network Expert
                    - Primary Task: Generate customized academic-style text attributes for specific categories in citation networks.
                    #### Capabilities
                    - Text Attribute Generation: Generate structured text attributes for given categories.
                    - Target Categories:
                        - Case_Based
                        - Genetic_Algorithms
                        - Neural_Networks
                        - Probabilistic_Methods
                        - Reinforcement_Learning
                        - Rule_Learning
                        - Theory
                    #### Instructions
                    Input:
                    - A category name from the list above.
                    - The number of papers to generate.
                    Output Format:
                    Generate the specified number of text attributes in JSON format to ensure structured output.
                    Each paper should have:
                    - Category: The research field of the paper.
                    - Title: A concise academic-style title (8-15 words).
                    - Abstract: A brief academic abstract (50-150 words).

                    Example Output:
                    [
                        {"Category": "Neural_Networks", "Title": "Advancements in Convolutional Neural Networks for Image Recognition", "Abstract": "This paper explores the recent developments in convolutional neural networks (CNNs), focusing on their applications in computer vision. We analyze various architectures, including ResNet and EfficientNet, and compare their performance on large-scale datasets."},
                        {"Category": "Reinforcement_Learning", "Title": "Deep Q-Learning for Autonomous Navigation", "Abstract": "This study investigates the application of Deep Q-Networks (DQN) in robotic navigation. We present experimental results demonstrating how reinforcement learning can enable agents to learn optimal navigation policies in dynamic environments."}
                    ]
                    - Ensure the JSON data you provide is not wrapped as a string and can be directly parsed with json.loads.
                    - Write in a formal and academic style.
                    - Ensure clarity and coherence in abstracts.
                    - Do not add any extra explanations, numbers, or special characters.
                        """
        pt = f"I want to generate {num} papers about {cate}."
    
  
    with open(f"{basepath}/data/{dataset}/processed/LLMdata.pkl", "rb") as f:
        classesAll = pickle.load(f)
    f.close()
    
    if len(classesAll[cate]) > num:
        classesList = classesAll[cate][:num]
    else:
        classesList = classesAll[cate][:-1]

    # Convert texts into embeddings
    embedding = text_tor(device,classesList, model, tokenizer).last_hidden_state[:, 0, :]
    
    return embedding,len(classesList)

def get_cateA(cata,dataset):
    """
       Return the textual name of a category ID for a given dataset.

       Args:
           cata (int): Index of the category/class.
           dataset (str): Name of the dataset (e.g., "cora", "citeseer", "wiki-cs", "pubmed").

       Returns:
           str: Corresponding category name.
       """
    if dataset == "cora": 
        coraCate = ["Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
                    "Reinforcement_Learning", "Rule_Learning", "Theory"]
        return coraCate[cata]
    elif dataset == "citeseer": 
        citeseerCate = ["Agents", "AI", "DB", "IR",
                        "ML",
                        "HCI"]
        return citeseerCate[cata]
    elif dataset == "wiki-cs": 
        wikiCate = ["Artificial Intelligence", "Computer Vision", "Cybersecurity", "Data Science", "Databases",
                    "Hardware & Architecture", "Human-Computer Interaction", "Machine Learning",
                    "Natural Language Processing", "Theoretical Computer Science"]
        return wikiCate[cata]
    elif dataset == "pubmed":
        pubmed = ["Experimental Diabetes Mellitus", "Type 1 Diabetes Mellitus", "Type 2 Diabetes Mellitus"]
        return pubmed[cata]



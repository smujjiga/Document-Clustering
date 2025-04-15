from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def load_20newsgroups(categories:list|None=None, clean:bool=False, limit_to:int=-1, random_state:int=3):
    remove=("headers", "footers", "quotes") if clean else ()
    dataset = fetch_20newsgroups(    
        remove=remove,
        subset="all",
        categories=categories,
        shuffle=True,        
        random_state=random_state,
    )
    
    df = pd.DataFrame({
        "document": dataset.data,
        "label": dataset.target
    }).head(limit_to)

    return df


def load_kaggle_data(categories:list|None=None, clean:bool=False, limit_to:int=-1, random_state:int=3):
    df = pd.read_csv("data/df_file.csv")    
    df = df.rename(columns={
        "Text": "document",
        "Label": "label"
    }).head(limit_to)
    return df


def load_quora_qa_data(categories:list|None=None, clean:bool=False, limit_to:int=-1, random_state:int=3):
    df = pd.read_json("hf://datasets/toughdata/quora-question-answer-dataset/Quora-QuAD.jsonl", lines=True)
    df = df.rename(columns={
        "question": "document",
    }).head(limit_to)
    return df


def plot_K(X):

    WCSS=[]
    for i in range(1,510, 10):
        kmeans=KMeans(n_clusters=i,init='k-means++')
        kmeans.fit(X)
        WCSS.append(kmeans.inertia_)
        
    plt.close()
    plt.plot(range(1,510, 10),WCSS)
    plt.savefig("images/Eblow.png")
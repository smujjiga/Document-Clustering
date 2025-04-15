import json
import logging
import os
import random
import textwrap
from collections import Counter, defaultdict
import plotly.express as px
import faiss
    
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import scipy
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from collections import defaultdict
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics


logging.basicConfig(level=logging.INFO)


class Clusters:
    def __init__(
        self,             
        embeddings:dict,
        dim_reduction:dict|None,
        clustering:dict|None,
        vizualize:bool=True,
        exp_name:str="No Name"   
    ):
        self.exp_name = exp_name
        self.embeddings = embeddings
        self.dim_reduction = dim_reduction
        self.clustering = clustering
        self.vizualize = vizualize


    def vectorize(self, documents, embeddings):
        logging.info(f"Vectorizing using {embeddings['model']}")

        if embeddings["model"].lower() == "tfidf":
            vectorizer = TfidfVectorizer(**embeddings['config'])
            start_time = time()            
            X = vectorizer.fit_transform(documents)

        logging.info(f"vectorization done in {time() - start_time:.3f} s")
        logging.info(f"Size: {X.shape[0]}, n_features: {X.shape[1]}")
        if scipy.sparse.issparse(X):
            # Sparcity: No:of non-zero entries divided by the total number of elements
            logging.info(f"Sparcity: {X.nnz / np.prod(X.shape):.3f}")


        if self.vizualize:
            x = X[:100]
            if scipy.sparse.issparse(x):            
                x = x.toarray()
            x = x[:,:1000]
            # import pdb; pdb.set_trace()
            fig = go.Figure(data=go.Heatmap(z=x))
            # fig = px.imshow(, color_continuous_scale='RdBu_r', origin='lower')
            fig.write_image("images/embeddings.png")

        return X


    def dim_reduce(self, X, dim_reduction):
        logging.info(f"Dimensionality Reduction using {dim_reduction['model']}")

        if dim_reduction["model"].lower() == "svd":
            lsa = make_pipeline(TruncatedSVD(**dim_reduction['config']), Normalizer(copy=False))
            start_time = time()
            X_lsa = lsa.fit_transform(X)
            explained_variance = lsa[0].explained_variance_ratio_.sum()
            
            logging.info(f"SVD done in {time() - start_time:.3f} s")
            logging.info(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")


            if self.vizualize:
                x = X_lsa[:100]
                if scipy.sparse.issparse(x):            
                    x = x.toarray()
                x = x[:,:1000]
                fig = go.Figure(data=go.Heatmap(z=x))                
                fig.write_image("images/svd_embeddings.png")

            return X_lsa
        
        

    def cluster_kmeans(self, kmeans, X, clustering, labels, evaluations, evaluations_std):
        name = self.exp_name

        logging.info(f"Clustering using {clustering['model']}")
       
        if not clustering.get("evaluate", False):
            kmeans.fit(X)
            cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
            return kmeans, cluster_ids, cluster_sizes
        else:
            train_times = []
            scores = defaultdict(list)
            for seed in range(5):
                kmeans.set_params(random_state=seed)        
                start_time = time()
                kmeans.fit(X)
                train_times.append(time() - start_time)
                scores["Homogeneity"].append(metrics.homogeneity_score(labels, kmeans.labels_))
                scores["Completeness"].append(metrics.completeness_score(labels, kmeans.labels_))
                scores["V-measure"].append(metrics.v_measure_score(labels, kmeans.labels_))
                scores["Adjusted Rand-Index"].append(
                metrics.adjusted_rand_score(labels, kmeans.labels_)
            )
            scores["Silhouette Coefficient"].append(
                metrics.silhouette_score(X, kmeans.labels_, sample_size=2000)
            )
            train_times = np.asarray(train_times)

            logging.info(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")  
            evaluation = {
               "estimator": name,
                "train_time": train_times.mean(),                
            }
            evaluation_std = {
                "estimator": name,
                "train_time": train_times.std(),
            }         

            for score_name, score_values in scores.items():
                mean_score, std_score = np.mean(score_values), np.std(score_values)
                logging.info(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}") 
                evaluation[score_name] = mean_score
                evaluation_std[score_name] = std_score               
            evaluations.append(evaluation)
            evaluations_std.append(evaluation_std)

    def cluster(self, X, clustering, labels=None, evaluations:list|None=None, evaluations_std:list|None=None):
        if clustering["model"].lower() == "kmeans":
            kmeans = KMeans(**clustering['config'])
            return self.cluster_kmeans(kmeans, X, clustering,labels,evaluations, evaluations_std)            
        elif clustering["model"].lower() == "minibatchkmeans":
            kmeans = MiniBatchKMeans(**clustering['config'])
            return self.cluster_kmeans(kmeans, X, clustering,labels,evaluations, evaluations_std)            

    def fit(self, documents:list[str], labels:list[int]|None=None, 
            evaluations:list|None=None, evaluations_std:list|None=None):
        self.documents = documents
        self.X = self.vectorize(documents=self.documents, embeddings=self.embeddings)

        if self.dim_reduction:
            self.X = self.dim_reduce(self.X, self.dim_reduction)

        if self.clustering:
            return self.cluster(self.X, self.clustering, labels, evaluations, evaluations_std)
        

def demo_20newsgroups():
    from clustering import Clusters
    import utils

    evaluations = []
    evaluations_std = []

    data = utils.load_20newsgroups()
    df = data.groupby(['label']).apply('count').reset_index()

    fig = px.bar(df, x="label", y="document", title="20 NewsGroups Dataset")
    fig.write_image("images/fig1.png")
    
    logging.info(data['document'].head())

    # # 1. With out cleaning and Without Dim reductions

    # clusters = Clusters(        
    #     embeddings={
    #         "model":  "tfidf",
    #         "config": {
    #             "max_df":0.5,   # ignoring terms that appear in more than 50% of the documents 
    #             "min_df":5, # ignoring terms that appear in less than 5 documents
    #             "stop_words":"english",
    #             },
    #         },
    #     dim_reduction=None,
    #     clustering=None,
    #     )
    
    # clusters.fit(data['document'].to_list())
    
    # # 2. Kmeans depends on the cluster centroids initialize  
    # for i in range(5):
    #     _, cluster_ids, cluster_sizes = clusters.cluster(clusters.X, clustering={
    #         "model": "KMeans",
    #         "config" : {
    #             "n_clusters":20,
    #             "max_iter":100,
    #             "n_init":1,
    #             "random_state":i,
    #         }})         
    #     logging.info(f"Cluster Sizes: {cluster_sizes}")

    # _, category_sizes = np.unique(data['label'], return_counts=True)
    # logging.info(f"True Cluster Sizes: {category_sizes}")

    # # How to choose best initilization >    
    # # Homogeneity: 0.349 ± 0.010
    # # Completeness: 0.398 ± 0.009
    # # V-measure: 0.372 ± 0.009
    # # Silhouette Coefficient: 0.007 ± 0.000

    # # 3. fit and eval
    # clusters = Clusters(        
    #     embeddings={
    #         "model":  "tfidf",
    #         "config": {
    #             "max_df":0.5,   # ignoring terms that appear in more than 50% of the documents 
    #             "min_df":5, # ignoring terms that appear in less than 5 documents
    #             "stop_words":"english",
    #             },
    #         },
    #     dim_reduction=None,
    #     clustering={
    #         "model": "KMeans",
    #         "config" : {
    #             "n_clusters":20,
    #             "max_iter":100,
    #             "n_init":1,
    #         },
    #         "evaluate": True,
    #     }
    # )

    # clusters.fit(data['document'].to_list(), data['label'].to_list())


    # # 4. Dim reduction using SVD

    # clusters = Clusters(        
    #     embeddings={
    #         "model":  "tfidf",
    #         "config": {
    #             "max_df":0.5,   # ignoring terms that appear in more than 50% of the documents 
    #             "min_df":5, # ignoring terms that appear in less than 5 documents
    #             "stop_words":"english",
    #             },
    #         },
    #     dim_reduction={
    #         "model": "svd",
    #         "config" : {
    #             "n_components":100
    #             },
    #     },
    #     clustering=None,
    # )

    # clusters.fit(data['document'].to_list())


    # # 5. Putting it all together with kmeans
    # clusters = Clusters(        
    # embeddings={
    #     "model":  "tfidf",
    #     "config": {
    #         "max_df":0.5,   # ignoring terms that appear in more than 50% of the documents 
    #         "min_df":5, # ignoring terms that appear in less than 5 documents
    #         "stop_words":"english",
    #         },
    #     },
    # dim_reduction={
    #     "model": "svd",
    #     "config" : {
    #         "n_components":100
    #         },
    # },
    #     clustering={
    #         "model": "KMeans",
    #         "config" : {
    #             "n_clusters":20,
    #             "max_iter":100,
    #             "n_init":1,
    #         },
    #         "evaluate": True,
    #     }
    # )

    # clusters.fit(data['document'].to_list(), data['label'].to_list())




    # 6. Putting it all together 
    Clusters(
        exp_name = "KMeans with TfidfVectorizer",
    embeddings={
        "model":  "tfidf",
        "config": {
            "max_df":0.5,   # ignoring terms that appear in more than 50% of the documents 
            "min_df":5, # ignoring terms that appear in less than 5 documents
            "stop_words":"english",
            },
        },
        dim_reduction=None,
        clustering={
            "model": "KMeans",
            "config" : {
                "n_clusters":20,
                "max_iter":100,
                "n_init":1,
            },
            "evaluate": True,
        }
    ).fit(
        data['document'].to_list(),
        data['label'].to_list(),
        evaluations,
        evaluations_std
        )

    Clusters(       
        exp_name = "KMeans with SVD", 
    embeddings={
        "model":  "tfidf",
        "config": {
            "max_df":0.5,   # ignoring terms that appear in more than 50% of the documents 
            "min_df":5, # ignoring terms that appear in less than 5 documents
            "stop_words":"english",
            },
        },
            dim_reduction={
        "model": "svd",
        "config" : {
            "n_components":100
            },
    },
        clustering={
            "model": "KMeans",
            "config" : {
                "n_clusters":20,
                "max_iter":100,
                "n_init":1,
            },
            "evaluate": True,
        }
    ).fit(
        data['document'].to_list(),
        data['label'].to_list(),
        evaluations,
        evaluations_std
        )



    Clusters(   
        exp_name = "MiniBatchKMeans with SVD",      
    embeddings={
        "model":  "tfidf",
        "config": {
            "max_df":0.5,   # ignoring terms that appear in more than 50% of the documents 
            "min_df":5, # ignoring terms that appear in less than 5 documents
            "stop_words":"english",
            },
        },
    dim_reduction={
        "model": "svd",
        "config" : {
            "n_components":100
            },
    },
        clustering={
            "model": "MiniBatchKMeans",
            "config" : {
                "n_clusters":20,
                "init_size":1000,
                "batch_size":1000,
                "n_init":1,
            },
            "evaluate": True,
        }
    ).fit(
        data['document'].to_list(),
        data['label'].to_list(),
        evaluations,
        evaluations_std
        )


    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 6), sharey=True)


    df = pd.DataFrame(evaluations[::-1]).set_index("estimator")
    df_std = pd.DataFrame(evaluations_std[::-1]).set_index("estimator")


    df.drop(
        ["train_time"],
        axis="columns",
    ).plot.barh(ax=ax0, xerr=df_std,colormap='Paired')
    ax0.set_xlabel("Clustering scores")
    ax0.set_ylabel("")

    df["train_time"].plot.barh(ax=ax1, xerr=df_std["train_time"],colormap='Paired')
    ax1.set_xlabel("Clustering time (s)")
    plt.tight_layout()

    plt.savefig("images/comparision.png")

if __name__ == "__main__":
    demo_20newsgroups()

import json
import numpy as np
import gensim.downloader as api
from sklearn.cluster import KMeans
import networkx as nx
import pickle
from collections import defaultdict
from pyvis.network import Network
import networkx as nx

class GraphHandler:
    def __init__(self, word2vec_model):
        self.word2vec = word2vec_model


    def load_reviews(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def generate_embeddings(self, reviews):
        embeddings = []
        rkp_labels = []  # corresponding RKPs
        rkp_ratings = []  # corresponding ratings for each RKP
        sentiments = []   # corresponding sentiments for each RKP
        for review in reviews:
            review_rating = review['rating']
            for key_point, sentiment in zip(review['review_key_points'], review['sentiments']):
                words = key_point.split()
                word_vectors = [self.word2vec[word] for word in words if word in self.word2vec]
                if word_vectors:  # we only include RKPs with valid embeddings
                    embeddings.append(np.mean(word_vectors, axis=0))
                    rkp_labels.append(key_point)
                    rkp_ratings.append(review_rating)  # the rating from the review
                    sentiments.append(sentiment)
        embeddings = np.array(embeddings)
        return embeddings, rkp_labels, rkp_ratings, sentiments





    def perform_clustering(self, embeddings, n_clusters):
        print(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters.")  # Debug
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)  # ensure 2D shape for single embedding
        if len(embeddings) < n_clusters:
            print(f"Warning: Not enough embeddings ({len(embeddings)}) for the number of clusters ({n_clusters}).")
            n_clusters = max(1, len(embeddings))  # at least one cluster

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels
    

    def cluster_embeddings(self, embeddings, n_clusters):
        if embeddings:
            embeddings = np.array(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(embeddings)
        return np.array([])


    def perform_sentiment_aware_clustering(self, embeddings, rkp_labels, sentiments, total_clusters):
        positive_embeddings = [emb for emb, sentiment in zip(embeddings, sentiments) if sentiment > 0]
        negative_embeddings = [emb for emb, sentiment in zip(embeddings, sentiments) if sentiment <= 0]

        # calculate the proportion of clusters for each sentiment
        total_positive = len(positive_embeddings)
        total_negative = len(negative_embeddings)
        total = total_positive + total_negative
        clusters_positive = max(1, int((total_positive / total) * total_clusters))
        clusters_negative = total_clusters - clusters_positive  # Remaining clusters are for negative sentiments

        # perform clustering for each sentiment group
        positive_labels = self.cluster_embeddings(positive_embeddings, clusters_positive)
        negative_labels = self.cluster_embeddings(negative_embeddings, clusters_negative)

        # convert NumPy arrays to lists
        positive_labels = positive_labels.tolist()
        negative_labels = negative_labels.tolist()

        # combine results with offsetting negative labels by positive cluster count
        labels = []
        offset = clusters_positive
        for sentiment in sentiments:
            if sentiment > 0:
                labels.append(positive_labels.pop(0))
            else:
                labels.append(negative_labels.pop(0) + offset)

        return labels


    def construct_graph(self, reviews, embeddings_info, labels):
        embeddings, rkp_labels, rkp_ratings, sentiments = embeddings_info
        G = nx.Graph()
        G.add_node('product')

        # create cluster nodes
        for i in range(max(labels) + 1):
            cluster_node = f'cluster_{i}'
            G.add_node(cluster_node, average_embedding=None, average_rating=None)
            G.add_edge('product', cluster_node)

        # aggregate embeddings and ratings by cluster
        cluster_embeddings = defaultdict(list)
        cluster_ratings = defaultdict(list)

        # add RKP nodes and collect data for averages
        for idx, label in enumerate(labels):
            rkp = rkp_labels[idx]
            embedding = embeddings[idx]
            rating = rkp_ratings[idx]
            rkp_node = f"{rkp} - Rating: {rating}"
            cluster_node = f'cluster_{label}'
            G.add_node(rkp_node, embedding=embedding, rating=rating)
            G.add_edge(cluster_node, rkp_node)

            # collect embeddings and ratings for averaging
            cluster_embeddings[label].append(embedding)
            cluster_ratings[label].append(rating)

        # calculate and set average embeddings and ratings for each cluster node
        for label in cluster_embeddings:
            if cluster_ratings[label]:  # Ensure there are ratings to process
                avg_embedding = np.mean(cluster_embeddings[label], axis=0)
                avg_rating = np.mean(cluster_ratings[label])
                G.nodes[f'cluster_{label}']['average_embedding'] = avg_embedding
                G.nodes[f'cluster_{label}']['average_rating'] = avg_rating

        return G


    def save_graph(self, graph, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)

    

    def save_clusters(self, rkp_labels, labels, output_path):
        from collections import defaultdict
        cluster_map = defaultdict(list)
        for rkp, label in zip(rkp_labels, labels):
            cluster_map[label].append(rkp)

        with open(output_path, 'w') as file:
            for label in sorted(cluster_map):  # sort clusters by their labels
                rkps = ', '.join(cluster_map[label])  # keep duplicates as they are
                file.write(f"Cluster {label}: {rkps}\n")



    def save_clusters_without_duplicates(self, rkp_labels, labels, output_path):
        from collections import defaultdict
        cluster_map = defaultdict(list)
        for rkp, label in zip(rkp_labels, labels):
            cluster_map[label].append(rkp)

        with open(output_path, 'w') as file:
            for label in sorted(cluster_map):  # clusters are now saved in sorted order
                rkps = ', '.join(set(cluster_map[label]))  # this line removes duplicates
                file.write(f"Cluster {label}: {rkps}\n")


    def visualize_graph(self, G, output_path):
            net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
            net.barnes_hut()

            for node, attr in G.nodes(data=True):
                size = 15 if "cluster" not in node else 30  # smaller size for RKPs, larger for clusters
                color = "#f0a30a" if node == "product" else "#7BE141" if 'positive' in node.lower() else "#FFC0CB"
                
                # default title information
                title = f"Node: {node}"

                # Check if node is the product, RKP, or cluster and adjust display info accordingly
                if node == "product":

                    title += "\nCentral Product Node"  # Special title for the product node

                elif "cluster" in node:

                    # we show the average rating and embedding if available for clusters
                    avg_rating = attr.get('average_rating', 'N/A')
                    avg_embedding = attr.get('average_embedding', [])

                    if isinstance(avg_embedding, np.ndarray):
                        avg_embedding = avg_embedding.tolist()
                    
                    title = f"Cluster Node: {node}\nAverage Rating: {avg_rating}\nAverage Embedding: {avg_embedding}"
                
                else: # then it should be an rkp
                    # display individual rating and embedding for RKPs
                    rating = attr.get('rating', 'N/A')
                    embedding = attr.get('embedding', [])

                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    title = f"RKP Node: {node}\nRating: {rating}\nEmbedding: {embedding}"

                net.add_node(node, label=node, color=color, size=size, title=title)

            for source, target in G.edges():
                net.add_edge(source, target)

            # save the visualization to an html file
            net.save_graph(output_path)

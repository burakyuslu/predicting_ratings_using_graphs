import json
import numpy as np
from scipy.spatial.distance import cosine
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PredictionHandler:
    def __init__(self, graph, word2vec):
        self.graph = graph
        self.word2vec = word2vec

    def load_reviews(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def predict_review_rating(self, rkps):
        predicted_ratings = []

        for rkp in rkps:
            rkp_embedding = self.generate_embedding(rkp)
            
            closest_cluster, _ = min(
                ((node, data) for node, data in self.graph.nodes(data=True) if 'average_embedding' in data),
                key=lambda x: cosine(x[1]['average_embedding'], rkp_embedding) if rkp_embedding is not None else float('inf'),
                default=(None, {'average_rating': None})

            )
            if closest_cluster:
                predicted_ratings.append(self.graph.nodes[closest_cluster]['average_rating'])
        
        return round(np.mean(predicted_ratings)) if predicted_ratings else None

    def generate_embedding(self, rkp):
        words = rkp.split()

        embeddings = [self.word2vec[word] for word in words if word in self.word2vec]
        return np.mean(embeddings, axis=0) if embeddings else None



    def evaluate_predictions(self, reviews):
        
        true_labels = []
        predicted_labels = []
        results = []

        correct_predictions = 0  # Counter for correct predictions
        for i, review in enumerate(reviews):

            predicted_rating = self.predict_review_rating(review['review_key_points'])
            true_rating = review['rating']

            true_labels.append(true_rating)

            predicted_labels.append(predicted_rating)
            correct = 'Yes' if predicted_rating == true_rating else 'No'

            if correct == 'Yes':
                correct_predictions += 1  # increment if prediction is correct
            results.append((i, true_rating, predicted_rating, correct))
        
        total_reviews = len(reviews)
        accuracy = (correct_predictions / total_reviews) * 100 if total_reviews else 0 

        # precision, recall, and F1 score
        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
        f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)

        return results, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


    def save_results(self, results, metrics, filepath):
        with open(filepath, 'w') as f:
            for index, true, predicted, correct in results:
                f.write(f"Review {index}: True Rating: {true}, Predicted Rating: {predicted}, Correct: {correct}\n")
            f.write(f"\nAccuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"Precision: {metrics['precision']:.2f}\n")
            f.write(f"Recall: {metrics['recall']:.2f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.2f}\n")


    def map_rating_to_class(self, rating):
        if rating in [1, 2]:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        elif rating in [4, 5]:
            return 'positive'
        return 'neutral'  # default if unexpected rating data comes our way


    def evaluate_predictions_three_class(self, reviews):

        true_classes = []
        predicted_classes = []
        results = []

        for i, review in enumerate(reviews):
            predicted_rating = self.predict_review_rating(review['review_key_points'])
            true_rating = review['rating']

            # Map ratings to classes
            predicted_class = self.map_rating_to_class(predicted_rating)
            true_class = self.map_rating_to_class(true_rating)
            true_classes.append(true_class)
            predicted_classes.append(predicted_class)
            correct = 'Yes' if true_class == predicted_class else 'No'
            results.append((i, true_class, predicted_class, correct))

        # Calculating metrics
        accuracy = accuracy_score(true_classes, predicted_classes) * 100
        precision = precision_score(true_classes, predicted_classes, average='macro', zero_division=1)
        recall = recall_score(true_classes, predicted_classes, average='macro', zero_division=1)
        f1 = f1_score(true_classes, predicted_classes, average='macro', zero_division=1)
        return results, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
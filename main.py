from graph_handler import GraphHandler
from prediction_handler import PredictionHandler  # Ensure this is defined in another file or added here
import gensim.downloader as api

def main():
    word2vec_model = api.load('word2vec-google-news-300')  # Load the word vector model, we pass this to different handler down the road

    # Initialize GraphHandler with the Word2Vec model 
    graph_handler = GraphHandler(word2vec_model)

    # Load reviews from JSON
    input_json_path = 'graph_inputs/input_graph_construction.json'
    reviews = graph_handler.load_reviews(input_json_path)

    # Generate embeddings and labels for each RKP
    embeddings, rkp_labels, rkp_ratings, sentiments = graph_handler.generate_embeddings(reviews)

    # perform sentiment-aware clustering
    num_clusters = 7
    labels = graph_handler.perform_sentiment_aware_clustering(embeddings, rkp_labels, sentiments, num_clusters)

    # construct and save the graph
    output_graph_path = 'output/output_graph.gpickle'
    graph = graph_handler.construct_graph(reviews, (embeddings, rkp_labels, rkp_ratings, sentiments), labels)
    graph_handler.save_graph(graph, output_graph_path)

    # Save cluster information (mostly debug, but it is also something interesting by itself)
    output_cluster_path = 'output/clusters.txt'
    graph_handler.save_clusters(rkp_labels, labels, output_cluster_path)

    # Visualize the graph
    output_directory = 'output/graph_visualization.html'
    graph_handler.visualize_graph(graph, output_directory)

    # predict and evaluate ratings
    prediction_handler = PredictionHandler(graph, word2vec_model)

    prediction_input_json_path = 'graph_inputs/prediction_input.json'
    reviews_to_predict = graph_handler.load_reviews(prediction_input_json_path)
    
    prediction_results, accuracy = prediction_handler.evaluate_predictions(reviews_to_predict)
    prediction_handler.save_results(prediction_results, accuracy, 'output/results.txt')

    prediction_results, metrics = prediction_handler.evaluate_predictions_three_class(reviews_to_predict)
    prediction_handler.save_results(prediction_results, metrics, 'output/results_three_class.txt')

if __name__ == '__main__':
    main()

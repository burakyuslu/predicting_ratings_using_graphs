import json

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd


def add_predicted_sentiments(input_file_path, output_file_path = 'graph_formatted_data_complete.json'):
    # Load the tokenizer and modelpath to the saved model.
    # Access to model folder. Place it together with this function: https://drive.google.com/drive/folders/1kJ0NQcxPTa-QJfteeqPQ4TJUGcSkZhDT?usp=sharing
    model_path = 'sentiment_model_completed'  
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model_eval = RobertaForSequenceClassification.from_pretrained(model_path,num_labels=2 )
    
    # Function to make predictions
    def predict_sentiment(phrase, context):
        inputs = tokenizer(phrase, context, return_tensors='pt', truncation=True, padding='max_length', max_length=500)
        with torch.no_grad():
            outputs = model_eval(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        #print(predicted_class_id)
        return predicted_class_id
        #return "positive" if predicted_class_id == 1 else "negative"

    # Load the input JSON file
    with open(input_file_path, 'r') as file:
        reviews = json.load(file)

    # Convert the input to the desired output format
    output_reviews = []
    for review in reviews:
        review_text = review["review_text"]
        key_points = review["review_key_points"]
        rating = review["rating"]
        #print(key_points)
        sentiments =[]
        for kp in key_points:
            sentiment = predict_sentiment(kp, review_text)
            sentiments.append(sentiment)
        
        # Add the sentiment to the review dictionary
        review["sentiments"] = sentiments
        
        # Append the updated review to the output list
        output_reviews.append(review)

    # Save the output JSON file
    with open(output_file_path, 'w') as file:
        json.dump(output_reviews, file, indent=4)
    
    print("Sentiments added")

    return(output_reviews)

add_predicted_sentiments('graph_formatted_data.json')
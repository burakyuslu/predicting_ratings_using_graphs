import gc
import gzip
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import pandas as pd

import requests
from openai import OpenAI

# This script fetches the reviews for all categories from the dataset at https://amazon-reviews-2023.github.io/
# To train our model we will only use a subset of this data, e.g., ~250 per category and 1000~ of "Unknown"

# To acquire labels for our training data, we will use OpenAI's API and further processing.

BASE_LINK = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/'
BASE_LINK_META = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/'
FILE_ENDING = '.jsonl.gz'
CATEGORIES = ['All_Beauty', 'Amazon_Fashion', 'Appliances', 'Arts_Crafts_and_Sewing', 'Automotive', 'Baby_Products',
              'Beauty_and_Personal_Care', 'Books', 'CDs_and_Vinyl', 'Cell_Phones_and_Accessories',
              'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Electronics', 'Gift_Cards', 'Grocery_and_Gourmet_Food',
              'Handmade_Products', 'Health_and_Household', 'Health_and_Personal_Care', 'Home_and_Kitchen',
              'Industrial_and_Scientific', 'Kindle_Store', 'Magazine_Subscriptions', 'Movies_and_TV',
              'Musical_Instruments', 'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Software',
              'Sports_and_Outdoors', 'Subscription_Boxes', 'Tools_and_Home_Improvement', 'Toys_and_Games',
              'Video_Games', 'Unknown']

RAW_DATA_DIRECTORY = 'dataset/raw_data/'
META_DATA_DIRECTORY = 'dataset/meta_data/'
SUBSET_DATA_DIRECTORY = 'dataset/subset_data/'
UNPACKED_DATA_DIRECTORY = 'dataset/raw_data_unpacked/'
UNPACKED_META_DATA_DIRECTORY = 'dataset/meta_data_unpacked/'
DATASET_FILE_PATH = 'dataset/dataset.json'

client = OpenAI()


def fetch_dataset_for_category(category: str):
    file_name = category + FILE_ENDING
    file_path = os.path.join(RAW_DATA_DIRECTORY, file_name)

    if not os.path.isfile(file_path):
        dataset_url = BASE_LINK + file_name
        print(f"Downloading {file_name} to {file_path}")
        response = requests.get(dataset_url, stream=True)

        if response.status_code == 200:
            os.makedirs(RAW_DATA_DIRECTORY, exist_ok=True)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {file_name} to {file_path}")
        else:
            print(f"Failed to download {file_name}. HTTP Status Code: {response.status_code}")
    else:
        print(f"{file_name} already exists at {file_path}")


def fetch_metadata_for_category(category: str):
    file_name = 'meta_' + category + FILE_ENDING
    file_path = os.path.join(META_DATA_DIRECTORY, file_name)

    if not os.path.isfile(file_path):
        dataset_url = BASE_LINK_META + file_name
        print(f"Downloading {file_name} to {file_path}")
        response = requests.get(dataset_url, stream=True)

        if response.status_code == 200:
            os.makedirs(RAW_DATA_DIRECTORY, exist_ok=True)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {file_name} to {file_path}")
        else:
            print(f"Failed to download {file_name}. HTTP Status Code: {response.status_code}")
    else:
        print(f"{file_name} already exists at {file_path}")


def get_product_name(parent_asin: str, category: str):
    file_path = os.path.join(UNPACKED_META_DATA_DIRECTORY, category + '.jsonl')
    with open(file_path, 'r', encoding='utf-8') as meta_file:
        for current_line_index, line in enumerate(meta_file):
            record = json.loads(line)
            if record['parent_asin'] == parent_asin:
                return record['title']
    return ''


def extract_subset_from_category(subset_size: int, seed: int, category: str, min_appearances: int,
                                 chunk_size: int = 10000):
    """This function works on the data extracted from the original Amazon reviews."""
    start_time = time.time()

    file_name = category + FILE_ENDING
    file_path = os.path.join(RAW_DATA_DIRECTORY, file_name)
    unpacked_file_path = os.path.join(UNPACKED_DATA_DIRECTORY, f"{category}.jsonl")

    print(f"Unzipping {category}...")

    # Ensure the unpacked directory exists
    if not os.path.exists(UNPACKED_DATA_DIRECTORY):
        os.makedirs(UNPACKED_DATA_DIRECTORY)

    # Step 1: Unzip the file to the unpacked directory
    with gzip.open(file_path, 'rb') as f_in:
        with open(unpacked_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Step 2: Collect ASINs that meet the min_appearances condition (Pass 1)
    asin_counts = {}

    for chunk in pd.read_json(unpacked_file_path, lines=True, chunksize=chunk_size):
        for asin, count in chunk['parent_asin'].value_counts().items():
            if asin in asin_counts:
                asin_counts[asin] += count
            else:
                asin_counts[asin] = count

    # Filter ASINs that meet the min_appearances requirement
    frequent_asins = [asin for asin, count in asin_counts.items() if count >= min_appearances]

    if len(frequent_asins) == 0:
        print(f"Not enough asins with more than {min_appearances} occurrences in {category}.")
        os.remove(unpacked_file_path)
        return

    # Determine the number of ASINs needed
    min_no_of_asins = subset_size // min_appearances

    # Shuffle and sample the ASINs
    selected_asins = pd.Series(frequent_asins).sample(n=min_no_of_asins, random_state=seed).tolist()

    # Step 3: Collect the rows corresponding to the sampled ASINs (Pass 2)
    accumulated_rows = 0
    df_subset_list = []

    for chunk in pd.read_json(unpacked_file_path, lines=True, chunksize=chunk_size):
        # Filter the chunk for rows with the selected ASINs
        asin_subset = chunk[chunk['parent_asin'].isin(selected_asins)]

        # Add the rows of the current ASIN subset to the final dataset
        df_subset_list.append(asin_subset)
        accumulated_rows += len(asin_subset)

        # Stop when we have enough rows
        if accumulated_rows > subset_size:
            break

    # Concatenate all the chunks that match the selected ASINs
    df_subset = pd.concat(df_subset_list)

    # If we have fewer rows than needed, print a warning
    if accumulated_rows < subset_size:
        print(
            f"Not enough rows to fulfill the subset size {subset_size} for {category}. Only {accumulated_rows} rows available.")

    # Remove duplicates
    df_subset = df_subset.drop_duplicates(subset=['parent_asin', 'timestamp', 'rating'])

    # Add category and review columns
    df_subset.loc[:, 'category'] = category
    df_subset.loc[:, 'review'] = np.where(
        df_subset['title'].str.endswith('.') | df_subset['title'].str.endswith('!'),
        df_subset['title'] + ' ' + df_subset['text'],
        df_subset['title'] + '. ' + df_subset['text']
    )

    # Step 4: Export to CSV
    columns_to_export = ['rating', 'parent_asin', 'timestamp', 'review', 'category']
    output_file_path = os.path.join(SUBSET_DATA_DIRECTORY, f"{category}.csv")
    df_subset.to_csv(output_file_path, columns=columns_to_export, index=False)

    # Clear memory by deleting large variables and calling garbage collection
    del df_subset, df_subset_list, asin_counts, frequent_asins, selected_asins, asin_subset
    gc.collect()

    # Step 5: Delete the unzipped file after processing
    if os.path.exists(unpacked_file_path):
        os.remove(unpacked_file_path)

    end_time = time.time()  # Stop the timer
    processing_time = end_time - start_time  # Calculate the processing time

    print(f"Processed {category} with {accumulated_rows} rows in {processing_time:.2f} seconds!")


def extract_subsets(subset_size: int, min_appearances: int):
    """This function works on the data extracted from the original Amazon reviews."""
    for category in CATEGORIES:
        subset_file_path = os.path.join(SUBSET_DATA_DIRECTORY, category + '.csv')
        subset_factor = 4 if category == 'Unknown' else 1
        if not os.path.isfile(subset_file_path):
            extract_subset_from_category(subset_size * subset_factor, 42, category, min_appearances)


def combine_subsets():
    """This function works on the data extracted from the original Amazon reviews."""
    df = pd.DataFrame()
    for category in CATEGORIES:
        subset_file_path = os.path.join(SUBSET_DATA_DIRECTORY, category + '.csv')
        df_subset = pd.read_csv(subset_file_path)
        df = pd.concat([df, df_subset])
    output_file_path = os.path.join('dataset/combined_dataset.csv')
    df.to_csv(output_file_path, index=False)


def sample_for_qa(n: int):
    """This function works on the data extracted from the original Amazon reviews."""
    df = pd.read_csv('dataset/combined_dataset.csv')
    sampled_df = df.sample(n=n, random_state=42)
    sampled_df.to_json('dataset/qa_dataset_unannotated.json', orient="records", indent=2)


def generate_dataset(samples: int, subset_size: int, min_appearances: int):
    extract_subsets(subset_size, min_appearances)
    combine_subsets()
    sample_for_qa(samples)


def prepare_dataset_for_qa(input_path: str, output_path: str):
    """This function works on the annotated dataset."""
    # Load the existing JSON data
    with open(input_path, 'r') as file:
        data = json.load(file)

    squad_formatted_data = []

    # Iterate through each entry in the dataset
    for entry in data:
        review = entry['review']  # This is the context
        questions = entry['questions']
        answers = entry['answers']

        for i, question in enumerate(questions):
            # Ensure we don't go out of range in case of mismatch
            if i < len(answers):
                answer_text = answers[i]['answer']
                # Find the correct answer start within the review (context)
                answer_start = review.find(answer_text)

                # Handle case where the answer was not found
                if answer_start == -1:
                    raise ValueError(f"Answer '{answer_text}' not found in review '{review}'")

                # Generate a random unique ID for the question
                question_id = str(uuid.uuid4())

                # Format in SQuAD style
                squad_entry = {
                    'id': question_id,  # Add the unique ID here
                    'answers': {
                        'answer_start': [answer_start],
                        'text': [answer_text]
                    },
                    'context': review,
                    'question': question,
                    'category': entry['category'],
                    'rating': entry['rating'],
                    'parent_asin': entry['parent_asin'],
                    'timestamp': entry['timestamp']
                }

                # Append to the new list
                squad_formatted_data.append(squad_entry)

    # Save the new formatted dataset
    with open(output_path, 'w') as output_file:
        json.dump(squad_formatted_data, output_file, indent=4)


def prepare_dataset_for_graph(input_path: str, output_path: str):
    """This function works on the annotated dataset."""
    # Load the existing JSON data
    with open(input_path, 'r') as file:
        data = json.load(file)

    graph_formatted_data = []

    for entry in data:
        review = entry['review']
        answers = entry['answers']
        rating = entry['rating']

        review_key_points = []

        for answer in answers:
            review_key_points.append(answer['answer'])

        graph_entry = {
            'review_text': review,
            'review_key_points': review_key_points,
            'rating': rating
        }

        graph_formatted_data.append(graph_entry)

    with open(output_path, 'w') as output_file:
        json.dump(graph_formatted_data, output_file, indent=4)


def prepare_dataset_for_product_aspects(input_path: str, output_path: str):
    # Load the dataset
    df = pd.read_csv(input_path)

    product_review_counts = df.groupby('parent_asin').filter(lambda x: len(x) >= 20)

    product_review_counts = product_review_counts.sample(frac=1, random_state=42)

    selected_reviews = []
    total_reviews = 0
    for product_id, product_reviews in product_review_counts.groupby('parent_asin'):
        selected_reviews.extend(product_reviews.to_dict(orient='records'))
        total_reviews += len(product_reviews)
        if total_reviews >= 200:
            break

    with open(output_path, 'w') as f:
        json.dump(selected_reviews, f, indent=4)


def prepare_data():
    prepare_dataset_for_qa('dataset/qa_dataset_annotated.json', 'dataset/qa_dataset_formatted.json')
    prepare_dataset_for_graph('dataset/qa_dataset_annotated.json', 'dataset/graph_dataset_formatted.json')
    prepare_dataset_for_product_aspects('dataset/combined_dataset.json',
                                        'dataset/product_aspects_dataset_formatted.json')


# prepare_dataset_for_qa('dataset/qa_dataset_annotated.json', 'dataset/qa_dataset_formatted.json')
# prepare_dataset_for_graph('dataset/qa_dataset_annotated.json', 'dataset/graph_dataset_formatted.json')
# extract_subsets(250, 20)
# combine_subsets()
# sample_for_qa(200)
prepare_dataset_for_product_aspects('dataset/combined_dataset.csv', 'dataset/product_aspects_dataset_formatted.json')

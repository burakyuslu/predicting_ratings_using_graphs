# Question and Answering System

This Python file handles all things related to the Question and Answering system.

* fine-tuning and evaluating the QA models in qa_eval_training/ (all notebooks in this folder are almost identical, where ML4NLP_qa_fine_tuning.ipynb is a DistillBERT model pre-trained on SQuAD and fine-tuned on our data, ML4NLP_qa_fine_tune_base_bert.ipynb is a non-pre-trained DistillBERT fine-tuned on our data, and ML4NLP_qa_eval_no_fine_tune.ipynb is a DistillBERT pre-trained on SQuAD without further fine-tuning)
* using Pattern Mining for extracting the product aspects of reviews in aspect_extraction_training_eval/
* ML4NLP_dataset_analysis.ipynb is a quick analysis of our dataset to have an overview over what data we use
* ML4NLP_combined.ipynb combines product aspect extraction and the QA model to retrieve the review key points in the review

Because review key point extraction is heavily reliant on the product aspect extraction, the overall performance is not very good. As mentioned in the project report, improving the performance of product aspect extraction should greatly improve the overall performance of this subsystem.

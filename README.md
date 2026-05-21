# MLOps Assignment 2 — Fine-tuning DistilBERT on Goodreads Dataset

## Overview
This project fine-tunes a DistilBERT model for multi-class book genre 
classification using the UCSD Goodreads dataset. The complete MLOps 
pipeline includes experiment tracking with Weights & Biases, GPU training 
on Kaggle, and model deployment on Hugging Face Hub.

## Setup Instructions

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Environment Variables
```bash
export WANDB_API_KEY=your_wandb_api_key
export HF_TOKEN=your_huggingface_token
```

### Run Training
Training is designed to run on Kaggle Notebook with GPU.
Open the Kaggle notebook link below and run all cells.

## Results

Epoch|  	Training Loss |	  Validation Loss |   	Accuracy   | 	F1
1    |   	2.543057      |	  2.389233        |  	  0.583125   | 	0.582607
2    |    1.955018      |	  2.249788        |  	  0.603750   | 	0.605045
3    |   	1.433714      |	  2.274478        |  	  0.608750   | 	0.611895



## Links
- Kaggle Notebook: https://www.kaggle.com/code/rashmikumari2509/mlops-assign2/
- Hugging Face Model: https://huggingface.co/Rashmii30/distilbert-goodreads-genres
- W&B Dashboard: https: https://wandb.ai/g25ait2083-iit/mlops-assignment2
  
## Model
- **Model:** DistilBERT (distilbert-base-cased)
- **Task:** Multi-class text classification (book genres)
- **Dataset:** UCSD Goodreads Reviews by Genre
- **Platform:** Kaggle (GPU T4 x2)

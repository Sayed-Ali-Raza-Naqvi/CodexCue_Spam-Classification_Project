# Spam Classification Project

## Overview
This project aims to classify SMS messages as either spam or ham (non-spam) using machine learning techniques. The dataset used for training and evaluation contains SMS messages labeled as spam or ham.

## Dataset
The dataset used in this project is the SMS Spam Collection dataset on Kaggle.

## Usage

### Data Cleaning
- Removed unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4).
- Dropped duplicate entries.
- Transformed target variable labels into numerical format (0 for ham, 1 for spam).
### Exploratory Data Analysis (EDA)
- Examined the distribution of spam and ham messages.
- Analyzed the characteristics of messages such as number of characters, words, and sentences.
- Visualized distributions and correlations using various plots and charts.
### Text Preprocessing
- Removed punctuation, stopwords, and performed stemming on text data.
- Generated word clouds to visualize the most common words in spam and ham messages.
### Machine Learning Model
- Utilized TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.
- Trained several Naive Bayes classifiers (Gaussian, Multinomial, Bernoulli) for spam classification.
- Evaluated model performance using accuracy, confusion matrix, and precision scores.
### Files
- spam.csv: The original dataset.
- model.pkl: Pickle file containing the trained Multinomial Naive Bayes model.
- vectorizer.pkl: Pickle file containing the TF-IDF vectorizer.
### Deployment
- Deployed the trained model using Streamlit for real-time sentiment analysis.

## Acknowledgments
- SMS Spam Collection Dataset from UCI Machine Learning on Kaggle
- Open-source community: Numerous open-source libraries and tools, including pandas, NumPy, scikit-learn, Matplotlib, seaborn, NLTK, and WordCloud.
- CampusX YouTube channel.
- Streamlit documentation

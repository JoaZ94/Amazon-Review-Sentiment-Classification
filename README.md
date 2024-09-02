
# Amazon Review Sentiment Classification

## 1. Overview

Source: https://www.kaggle.com/datasets/tarkkaanko/amazon?resource=download 

This project consists of rating the sentiment of shopper reviews.

Since the dataset does not contain a detailed explanation of its fields, an EDA was performed to determine the information provided by each field. The dataset contains Amazon product reviews and each row of the dataset contains information about a review, including:

- **num:** A unique identifier for each review.
- **reviewerName:** The name of the user who gave the review.
- **overall:** The overall rating the user gave the product (e.g. 5.0, 4.0, etc.).
- **reviewText:** The text of the review describing the user's experience with the product.
- **reviewTime:** The date the review was conducted.
- **day_diff:** The difference in days between the date of the review and the date on which any calculation related to the review was performed.
- **helpful_yes:** The number of votes that found the review helpful.
- **helpful_no:** The number of votes that found the review not helpful.
- **total_vote:** The total number of votes the review received.
- **score_pos_neg_diff:** The difference between helpful and unhelpful votes.
- **score_average_rating:** The weighted average rating that might reflect the usefulness of the review.
- **wilson_lower_bound:** A calculated value based on the rating and votes, used to calculate a more reliable rating based on the binomial distribution.

For this project, only the ‘overall’ and ‘reviewText’ fields were taken, where the aim is to generate a relationship between the qualitative variable (reviewText) and the quantitative variable (overall). To generate such a relationship, it is important to apply NLP techniques and sentiment analysis.

## 2. Methodology:

- EDA.
- Elimination of empty data.
- Generate sentiment based on the ‘overall’ field.
- Class balancing of the target variable (Oversampling).
- Text pre-processing:
  - Punctuation removal.
  - Conversion to lower case letters.
  - Tokenisation.
  - Removal of stopwords.
  - Lemmatisation
- Tokenisation.
- Sequence generation (numerical arrays).
- Padded sequences.
- Division of the dataset into training dataset and test dataset.
- Create, compile and train the neural network model with Tensorflow.
- Evaluate the model.
- Make new predictions.

## 3. General Objective:

Develop a sentiment analysis model using Natural Language Processing (NLP) techniques and machine learning to predict the polarity (positive, negative, or neutral) of Amazon product reviews based on the review text and the overall rating (overall).

## 4. Specific Objectives:

- Preprocess the review text: Apply NLP techniques such as tokenization, lemmatization, stopwords removal, and vectorization to convert textual reviews into a numerical format that can be used by machine learning models.

- Explore the relationship between ratings (overall) and review text (reviewText): Identify patterns and correlations between the given score and the sentiment expressed in the text.

- Build a classification model: Use machine learning algorithms and neural networks to classify reviews into sentiment categories based on the text and rating.

- Evaluate the accuracy and performance of the model: Measure the model's performance using metrics such as accuracy, recall, F1-score, and confusion matrix.

- Provide recommendations: Based on the analysis, suggest improvements in review management and interpretation by the company.

## 5. Hypotheses

### Hypothesis 1:

- H1: Reviews with higher ratings (overall of 4 or 5) tend to contain more positive language in the text (reviewText).

- Justification: It is expected that users who give higher ratings will use words with positive connotations, while lower ratings will be associated with more negative language.

### Hypothesis 2:

- H2: It is possible to predict the overall rating of a review with high accuracy based solely on the text (reviewText) using NLP techniques.

- Justification: Since the review text often reflects the user's experience, a well-trained model should be able to infer the rating the user gave to the product.

### Hypothesis 3:

- H3: The length and complexity of the review text (reviewText) influence the accuracy of sentiment analysis and rating prediction.

- Justification: Longer reviews may provide more context, which could improve the model's accuracy, but they might also introduce noise if they contain contradictory information.

## 5. Business Context

Sentiment analysis of Amazon reviews has significant commercial value for the company because:

1. Improving customer satisfaction: Understanding the sentiment behind reviews allows Amazon to identify problem areas and respond to customer concerns more proactively. For example, if a product receives many negative reviews, Amazon could contact the seller or manufacturer to improve the product's quality.

2. Optimizing product recommendations: Integrating sentiment analysis into the recommendation system could help Amazon suggest products that are well-received by customers, increasing the likelihood of purchase and reducing returns.

3. Monitoring product reputation: Amazon can use sentiment analysis to monitor the public perception of its products and services over time, identifying trends or changes in customer satisfaction.

4. Marketing segmentation: By segmenting customers based on their feelings and opinions expressed in reviews, Amazon can personalize its marketing strategies and promotions to target different customer groups more effectively.

5. Detecting fake reviews: Through sentiment analysis, it is possible to identify patterns that might indicate the presence of fake or manipulated reviews, helping to maintain the integrity of product reviews on the platform.

## 6. Development

Two notebooks were developed, sharing much of the code in common, where what varies is the type of classification. It was decided to make 2 independent notebooks due to the time it takes to train the neural network, therefore, by having 2 notebooks, they can be trained simultaneously:

1. Binary classification notebook, where the analysis of the reviews can be positive or negative.

2. Multi-class classification notebook, where the prediction of the reviews can be positive, neutral or negative.

## 7. Conclussion

- It can be concluded that LSTM neural networks in conjunction with NPL tools are a great combination for analyzing reviews and determining review sentiment.

- Furthermore, the capability of the neural network will depend on the 'corpus' (set of reviews) and the quality of the reviews.

- The elimination of stop works and the process of lemmatization allows to reduce the dictionary of words, making a more efficient use of resources and generalizing the prediction of reviews.

- For both cases, a resampling technique known as oversampling was used so that the frequency asymmetry of the possible assessments would not affect the neural network.

- Both classification models (Binary and Multiclass) performed well, obtaining scores above 98%, but the multiclass classification neural network performed slightly better, reaching an accuracy of 99.44%, in contrast to 98.48% for the binary classifier.
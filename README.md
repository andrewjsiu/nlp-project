# NLP: Analyzing Healthcare Reviews and Predicting Their Usefulness

## Table of Contents

1. Motivation
2. Text Preprocessing
3. Predicting Review Usefulness with Word2Vec Features
4. Predicting Review Usefulness with Doc2Vec Features

### Motivation

How much can we learn from healthcare reviews written by patients? Patients are no experts, but they do have the first-hand experience of a treatment and know exactly how they feel before and after the treatment. To go beyond the individual subjective experience, the community can also vote on the usefulness of a review. 

Can we use text alone to predict review usefulness? It often takes time for a good review to gather many votes that it deserves. But if useful reviews tend to use similar words or styles, we should be able to predict how many useful votes a review will eventually obtain as soon as it is written. Such model will help people to quickly find the most useful reviews and hopefully better care.

### Text Preprocessing

[Yelp dataset](https://www.yelp.com/dataset_challenge) contains more than 4 million reviews written by 1 million users for 144 thousand businesses. The data are provided in .json format as separate files for businesses and reviews. The files are text files (UTF-8) with one json object corresponding to an individual record in each line.

To prepare the data in a more usable format, each business record is converted to a Python dictionary. Each business dictionary has a business_id as a unique identifier and an array of relevant categories that the business belongs to. I first keep all businesses that are in the “Health & Medical” category and then remove these health & medical related businesses that also belong to “Restaurants”, “Food”, or “Pets”.  To focus on English text, I also remove businesses in Germany and create a final set of business IDs for all 10,211 healthcare entities. 

Using this set of healthcare IDs, I create a new file that contains only the text from reviews for these healthcare organizations. Each review is written as a line in the new file, resulting in a total of 114,556 healthcare reviews. 

Certain words often appear right next to each other as a phrase or meaningful concept. For instance, birth control should not be considered as two separate words but as one phrase birth_control. I can develop a phrase model by looping over all the words in the reviews and looking for words that tend to co-occur one after another, with a frequency that is much higher than what we would expect by random chance. The gensim library offers a Phrases class to detect common phrases from a stream of sentences. I run the phrase detection twice over the whole corpus to capture common phrases that are longer than two words such as american_dental_association. 

Before running the phrase modeling, I segment the reviews into sentences and use the spaCy library to lemmatize the text. After two passes of the phrase modeling, I remove all English stopwords and proper names and create a file of transformed text with one review per line. Now the data have two well-defined layers: documents for reviews and tokens for words and phrases. 

### Predicting Review Usefulness with Word2Vec Features

The main idea of word2vec is that we can learn something about the meaning of a word based on the context of the word. The context words that appear immediately before or after a center word can be used to predict what the center word might be. This is the goal of the continuous bag-of-words (CBOW) algorithm, which runs through the entire corpus with a sliding window using the surrounding words to predict the center word in each window. At the core of the word2vec model is to train a neural network that produces a vector representation for each word in the corpus. Words that share common contexts will have similar word vectors.  For instance, the words that share the most similar word vectors as the word ‘dentist’ are ‘pediatric_dentist’ and ‘orthodontist’.

For the healthcare reviews, I first build the vocabulary from going through the entire corpus and then train a word2vec model with 10 epochs. There is a total of 6,382 terms in the word2vec vocabulary. With a dictionary mapping each word to a 100-dimensional semantic vector, we can build features for each document. The simplest way is to average word vectors for all word in a review. Another version is to weight each word by the its TF-IDF. Some have also suggested using the maximum vector plus the minimum vector in a review.

The target variable is the number of useful votes a review receives. Since most of the reviews have zero useful votes and the distribution is highly skewed to the right, I take the logarithm of the number of useful votes plus one. To measure the predictive performance of a model, I use 5-fold cross-validation to generate an overall Root Mean Squared Error (RMSE) of the transformed target variable. 

I find that linear regression performs poorly in predicting usefulness with an overall RMSE of more than 175. Ridge regression that penalizes large coefficients to control for overfitting performs significantly better with an overall RMSE of 0.617 with or without TF-IDF weighting on the average word vectors. Random Forest regressor further lowers the RMSE to 0.613, but the best performer is the XGBoost regressor achieving a RMSE of 0.603. 

### Predicting Review Usefulness with Doc2Vec Features

A more direct way to use neural network to generate features for predictive modeling is to train a Doc2Vec model, which creates a vector representation for each document, paragraph or review. The reviews have variable lengths, but the trained vectors have a fixed length. The algorithm runs through the entire corpus the first time to build the vocabulary. To produce better results, I then iterate through the corpus 10 more times to learn a vector representation for each word and for each review. 

For every regression model, the predictive performance with Doc2Vec is better than that with features generated from Word2Vec. The linear regression with Doc2Vec achieves an overall RMSE of 0.597 with five folds of cross-validation. The best performer is still XGBoost regressor with a RMSE of 0.585. 

In practice, we may only have a small dataset, so I also check how predictive performance varies with the amount of training data. Unsurprisingly, RMSE falls as we use more training data from 30% to 70% of the entire corpus, but the decrease in error is less than 0.003 for all models. Thus, even if we only use 30% of the data, we can still achieve reasonably good out-of-sample prediction. 

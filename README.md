## Table of Contents

1. [Text Preprocessing](https://github.com/andrewjsiu/Capstone_Project_NLP/blob/master/README.md#text-preprocessing)
2. [Predicting Review Usefulness with Word2Vec Features](https://github.com/andrewjsiu/Capstone_Project_NLP/blob/master/README.md#predicting-review-usefulness-with-word2vec-features)
3. [Predicting Review Usefulness with Doc2Vec Features](https://github.com/andrewjsiu/Capstone_Project_NLP/blob/master/README.md#predicting-review-usefulness-with-doc2vec-features)

## NLP: Analyzing Healthcare Reviews and Predicting Review Usefulness

Yelp has gathered millions of reviews on various organizations, but not all reviews are equally useful. To measure usefulness, Yelp has asked the community to vote on the usefulness of each review. However, it often takes weeks or months for a good review to accumulate the votes it deserves. It would help people to find the most useful reviews more quickly if we can predict how useful a review would be as soon as it is written. 

The goal of this project is to build a predictive model based on the text alone, so there is no need to wait for people to vote and gather additional data.

### Text Preprocessing

[Yelp dataset](https://www.yelp.com/dataset_challenge) contains more than 4 million reviews written by 1 million users for 144 thousand businesses. The data are provided in .json format as separate files for businesses and reviews. The files are text files (UTF-8) with one json object corresponding to an individual record in each line.

To prepare the data in a more usable format, each business record is converted to a Python dictionary. Each business dictionary has a business_id as a unique identifier and an array of relevant categories that the business belongs to. I first keep all businesses that are in the “Health & Medical” category and then remove these health & medical related businesses that also belong to “Restaurants”, “Food”, or “Pets”.  To focus on English text, I also remove businesses in Germany and create a final set of business IDs for all 10,211 healthcare entities. 

Using this set of healthcare IDs, I create a new file that contains only the text from reviews for these healthcare organizations. Each review is written as a line in the new file, resulting in a total of 114,556 healthcare reviews. 

Certain words often appear right next to each other as a phrase or meaningful concept. For instance, birth control should not be considered as two separate words but as one phrase birth_control. I can develop a phrase model by looping over all the words in the reviews and looking for words that tend to co-occur one after another, with a frequency that is much higher than what we would expect by random chance. The gensim library offers a Phrases class to detect common phrases from a stream of sentences. I run the phrase detection twice over the whole corpus to capture common phrases that are longer than two words such as american_dental_association. 

Before running the phrase modeling, I segment the reviews into sentences and use the spaCy library to lemmatize the text. After two passes of the phrase modeling, I remove all English stopwords and proper names and create a file of transformed text with one review per line. Now the data have two well-defined layers: documents for reviews and tokens for words and phrases. 

### Predicting Review Usefulness with Word2Vec Features

One way to vectorize the text is count the frequency of different words used and rely on the word-word co-occurrence matrix, leveraging the global statistical information. But it tends to perform poorly on word analogy. Another method is based on local context windows, and the main idea is that we can learn something about the meaning of a word based on the context of the word. The context words that appear immediately before or after a center word can be used to predict what the center word might be. This is the goal of the continuous bag-of-words (CBOW) algorithm, which runs through the entire corpus with a sliding window using the surrounding words to predict the center word in each window. At the core of the word2vec model is to train a neural network that produces a vector representation for each word in the corpus. Words that share common contexts will have similar word vectors.  For instance, the words that share the most similar word vectors as the word ‘dentist’ are ‘pediatric_dentist’ and ‘orthodontist’.

For the healthcare reviews, I first build the vocabulary from going through the entire corpus and then train a word2vec model with 10 epochs. There is a total of 6,382 terms in the word2vec vocabulary. With a dictionary mapping each word to a 100-dimensional semantic vector, we can build features for each document. The simplest way is to average word vectors for all word in a review. Another version is to weight each word by the its TF-IDF. Some have also suggested using the maximum vector plus the minimum vector in a review.

The target variable is the number of useful votes a review receives. Since most of the reviews have zero useful votes and the distribution is highly skewed to the right, I take the logarithm of the number of useful votes plus one. To measure the predictive performance of a model, I use 5-fold cross-validation to generate an overall Root Mean Squared Error (RMSE) of the transformed target variable. 

I find that linear regression performs poorly in predicting usefulness with an overall RMSE of more than 175. Ridge regression that penalizes large coefficients to control for overfitting performs significantly better with an overall RMSE of 0.617 with or without TF-IDF weighting on the average word vectors. Random Forest regressor further lowers the RMSE to 0.613, but the best performer is the XGBoost regressor achieving a RMSE of 0.603. 

Since the document features are average word vectors, we could find the predicted usefulness of a document that contains a single word. This would allow us to find which words are most useful or least useful. I first estimate a regression model and then use it to predict the number of useful votes each single word vector will obtain. In ranking all words by their predicted usefulness, both XGBoost and Ridge regressors show that words that involve a dollar amount, 'price', 'co-pay' or 'cash' tend to be the most useful words, perhaps because they are informative and provide objective facts. Words like 'confidence', 'amazing' and 'excellent' are mere subjective feelings, so they are among the least useful of all words.

### Predicting Review Usefulness with Doc2Vec Features

A more direct way to use neural network to generate features for predictive modeling is to train a Doc2Vec model, which creates a vector representation for each document, paragraph or review. The reviews have variable lengths, but the trained vectors have a fixed length. The algorithm runs through the entire corpus the first time to build the vocabulary. To produce better results, I then iterate through the corpus 10 more times to learn a vector representation for each word and for each review. 

For every regression model, the predictive performance with Doc2Vec is better than that with features generated from Word2Vec. The linear regression with Doc2Vec achieves an overall RMSE of 0.597 with five folds of cross-validation. The best performer is still XGBoost regressor with a RMSE of 0.585. 

In practice, we may only have a small dataset, so I also check how predictive performance varies with the amount of training data. Unsurprisingly, RMSE falls as we use more training data from 30% to 70% of the entire corpus, but the decrease in error is less than 0.003 for all models. Thus, even if we only use 30% of the data, we can still achieve reasonably good out-of-sample prediction. 

To further improve the predictive performance, we can tune the several parameters of XGBoost regressor, such as the maximum depth of a tree which determines the complexity of the tree, subsample ratio of the training instance for growing trees, and the degree of regularization on weights. In the end, using the best values for these parameters found by a grid search cross validation the RMSE on the test set falls to 0.5821.  

### Conclusion

The goal of the project is to provide a model that can predict the usefulness of a review as soon as it is written. Then we can always show people the most useful reviews without having to wait for the community to vote on review usefulness. In this project, I have processed the text of Yelp reviews by lemmatizing the words, finding common phrases, removing English stopwords. I then trained the word2vec model to learn word embeddings by making predictions based on local context windows and the doc2vec model that assigns a vector for each document. The results show that the feature vectors obtained from the doc2vec model combined with the XGBoost estimator generated the highest predictive performance. 

There are several ways to further improve the analysis. First, instead of training word vectors based on narrow context windows, we can train global vectors for word representation (GloVe) by making use of the global word-to-word co-occurrence matrix. GloVe consists of a weighted least squares model that trains on such co-occurrence counts. Second, the results are likely to improve if we also include other features, such as the star rating, the text length and the number of reviews the reviewer has written. Lastly, I could try a newer and faster gradient boosting framework called the Light Gradient Boosting Machine (LightGBM), which finds the best split among all tree leaves instead of being constrained by the tree depth. 


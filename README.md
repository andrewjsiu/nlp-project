# Capstone Project: Analyzing Healthcare Reviews with NLP

## Introduction

It is difficult to find a good doctor who will always take the patient’s best interest as top priority. Although patients are no experts, they do have the first-hand experience of the treatment and know exactly how they feel before and after treatment. One way to judge the quality of a doctor’s diagnosis or treatment is to go directly to listen to what the patients would say. Yelp’s business dataset contains 10,211 different healthcare organizations in 11 cities across 4 countries and 114,556 reviews about them.

I propose to analyze Yelp reviews on hospitals, doctors, dentists and other health professionals to better understand what makes a healthcare practice excellent from the patients’ perspective. I will use latent Dirichlet allocation (LDA) to discover prominent topics among these one hundred thousand reviews on patients’ experiences. Such topic modelling will help us see the common themes that patients tend to compliment or complain about. I can use pyLDAvis library to visualize the topic model with an interactive interface to help explore the topics discovered. 

I will also train a word vector model with word2vec using gensim library. I can visualize the word vectors by reducing dimensionality with t-Distributed Stochastic Neighbor Embedding. This word2vec model can be used to produce automatic summary of all the reviews of a healthcare practice or to predict the positive or negative sentiment toward a practice in a review. 

In sum, the automatic summary of key points for each healthcare practice will make it easier for patients to find a good-quality doctor who genuinely cares for the patients. Insights on what patients tend to compliment or complain about will help healthcare practices improve their service by gaining a more thorough understanding what it is like to be at the receiving end of various treatments. 

## Data Wrangling

Yelp dataset contains more than 4 million reviews written by 1 million users for 144 thousand businesses. The data are provided in .json format as separate files for businesses and reviews. The files are text files (UTF-8) with one json object corresponding to an individual record in each line.

To prepare the data in a more usable format, I first read in each business record and convert it to a Python dictionary. Each business dictionary has a business_id as a unique identifier and an array of relevant categories that the business belongs to. Then I filter out businesses that are not in the “Health & Medical” category and create a set of business IDs for all 10,211 healthcare entities. 
Using this set of healthcare IDs, I create a new file that contains only the text from reviews for these healthcare organizations. Each review is written as a line in the new file, and there is a total of 114,556 healthcare reviews. 

Certain words often appear right next to each other as a phrase or meaningful concept. For instance, birth control should not be considered as two separate words but as one phrase. I can develop a phrase model by looping over all the words in the reviews and looking for words that tend to co-occur one after another much more frequently than what we would expect by random chance. The gensim library offers a Phrases class to detect common phrases from a stream of sentences. I run the phrase detection twice over the whole corpus to capture common phrases that are longer than two words. 

Before running the phrase modeling, I segment the reviews into sentences and use the spaCy library to lemmatize the text. After two passes of the phrase modeling, I remove all English stopwords and create a file of transformed text with one review per line. Now the data have two well-defined layers: documents for reviews and tokens for words and phrases. 

Next, I use Latent Dirichlet Allocation (LDA) to discover a set of “topics” as an intermediate layer.  The assumption is that each review is represented as a mixture of topics and that each topic is represented as a mixture of tokens in the vocabulary following the Dirichlet probability distributions.  The number of topics is a hyperparameter, and for the initial trial I will choose 30 topics. These topics and the associated probability distributions are discovered by an algorithm that maximizes the likelihood of observing the reviews in the corpus. The idea is to capture some latent structure and organization within these reviews.

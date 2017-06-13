# Capstone_Project_NLP

## Analyzing Healthcare Reviews

It is difficult to find a good doctor who will always take the patient’s best interest as top priority. Although patients are no experts, they do have the first-hand experience of the treatment and know exactly how they feel before and after treatment. One way to judge the quality of a doctor’s diagnosis or treatment is to go directly to listen to what the patients would say. Yelp’s business dataset contains 10,211 different healthcare organizations in 11 cities across 4 countries and 114,556 reviews about them.

I propose to analyze Yelp reviews on hospitals, doctors, dentists and other health professionals to better understand what makes a healthcare practice excellent from the patients’ perspective. I will use latent Dirichlet allocation (LDA) to discover prominent topics among these one hundred thousand reviews on patients’ experiences. Such topic modelling will help us see the common themes that patients tend to compliment or complain about. I can use pyLDAvis library to visualize the topic model with an interactive interface to help explore the topics discovered. 

I will also train a word vector model with word2vec using gensim library. I can visualize the word vectors by reducing dimensionality with t-Distributed Stochastic Neighbor Embedding. This word2vec model can be used to produce automatic summary of all the reviews of a healthcare practice or to predict the positive or negative sentiment toward a practice in a review. 

In sum, the automatic summary of key points for each healthcare practice will make it easier for patients to find a good-quality doctor who genuinely cares for the patients. Insights on what patients tend to compliment or complain about will help healthcare practices improve their service by gaining a more thorough understanding what it is like to be at the receiving end of various treatments. 

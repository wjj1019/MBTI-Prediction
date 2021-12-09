# MBTI-Prediction
In personality typology, the Myers–Briggs Type Indicator (MBTI) is an introspective self-report questionnaire indicating differing psychological preferences in how people perceive the world and make decisions.The test attempts to assign four categories: introversion or extraversion, sensing or intuition, thinking or feeling, judging or perceiving.
* For this study, we will focus on the performance therefore, look specifically for prediction of the first category (Prediction for Extraversion vs Introversion)


![ScreenShot](https://github.com/wjj1019/MBTI-Prediction/blob/main/Process/MyersBriggsTypes.png)


Project Purpose
-----------------
The aim of this project is to make a prediction of an individual's MBTI (myers briggs personality types) based on their comments made to a certain blog
The prediction can lead towards better identification of a person's trait and may lead to more enhanced Recommenders Systems.

Data
------
Kaggle Dataset
https://www.kaggle.com/datasnaek/mbti-type
- ~ 8000 rows of samples 
- Columns: Type (MBTI Type) , Posts (Comments from each users)

Methods
--------
1. Data Cleaning
- Text data was processed by removing unwanted/unecessary information (Such as Website Link)

2. Feature Engineering/ Text Preprocessing
- Removal of Stopwords and Punctuations
- All texts into lower case
- Links and Emails Dropped (Unnecessary Information)
- Words with less than two words dropped 
- Stemming and Lemmatizing 

3.  Feature Extraction
- Count Vectorizer (Bag of Words): Sparse Matrix with frequency measure for each word per document
- TF-IDF (Term Frequency - Inverse Document Frequency): Measure of importance of a word is to a document (Weighted measure - Score based representation)
- Word Embedding (Word2Vec): Pretrained embedding matrix (300 Dimension) represents each word in 300 dimensional space 
- Word Vector -> Sentence Vector: Word2Vec consists of 300dim vectors for each word, therefore avearging all the vectors (words) within a document to create a sentence vector
- Word2Vec + TFIDf: Combination of Word2Vec with TFIDF Score, Multiplying the TFIDF Score with Word Vector from Word2Vec and take the average to creat a sentence vector

4. Imbalance Dataset Control
- The data was heavily skewed towards one side and therefore performed Oversampling Method (SMOTE) to balance out the dataset
- Combination of Over and Undersampling can be done for better performance, however, the memory usage was limited in the local environment

4.  Machine Learning (Classification)
Input/Independent Variable(X): Sentence Vector (One of the 5 Methods from Feature Extraction Method)
Output/Dependent Variable (Y): Label Encoded MBTI Type (E or I => 0, 1)

- Logistic Regression
- Support Vector Machine (Lasso and Ridge Regularizied)
- Random Forest
- Multinomial Naive Bayes

Evaluation Method
- Accuracy Score
- Confusion Matrix 

5. Neural Network
- LSTM with Embedding Layer as an Input
- LSTM with Pretrained Word2Vec as an Input (Embedding Matrix)
- Multilayered Perceptron 
- LSTM + Convolutional Layer 

Evaluation Method
- Accuracy Score

Finding/Conclusion
-------------------
ML Performance
Performance of Random Forest along with Word2Vec + TF-IDF had the higest performance having 75% accuracy 
Performance of Logistic Regression, SVM (Ridge and Lasso) along with TF-IDF (SMOTE) had accuracy score around 73%

Neural Network Performance
10 Epochs with Batch size 34 applied to every model 
LSTM + Conv + Word2Vec Embedding had the best performance:
- Training Time faster than typical RNN 
- Training and Validation Accuracy Increase over Epochs 
- Training and Validation Loss Decrease over Epochs

Challenge
---------
Since the data was expanded to have over 300k samples, there was limitation to process the big data under local environment.
Pyspark can be utilize to performance parallel compuation
- In this notebook, I have sepearted the data into 8 identicall distributed datasets

Nueral Network Performance can be further analyze by performing hyperparameter tuning and increasing the epochs 



# Business Understanding :

Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster.

In this project , we will be building a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

Dataset is taken from Kaggle : https://www.kaggle.com/c/nlp-getting-started/data

# Main Libraries used in the study :

1. pandas
2. numpy
3. matplotlib
4. seaborn
5. sklearn

## Description of the files :

### ApplyClassificationModel.py :
  In this file, helper methods are created which fit and evaluate Decision Tree, Logistic Regression, SVM, Naive Bayes, KNN on the input training and test parameter and print the results.
  
### DataProcessing.py :
  In this file, data cleansing part is done. Helper methods are created which will :
  
  1. Normalize the tweets to the lower format.
  2. Remove tags like #, @ from the words in the text.
  3. Remove punctuations.
  4. Remove stop words like is, the, on etc.
  5. Remove words which are of length less than 2.
  6. Remove alpha-numeric string.
  7. Reduce the words to their root forms using Lemmatization

### RealOrNotDisasterTweets.ipynb :
  This is the main notebook which is going to call the helper methods and contains the complete analyisis of the study.
  
# Visualizations from the Notebook :

#### a. Target Distribution in Training Dataset

![image](https://user-images.githubusercontent.com/52653296/81777066-b579e700-950d-11ea-9891-b95fb2bd9b96.png)

#### b. Frequent Hashtags in Training Set

![image](https://user-images.githubusercontent.com/52653296/81777117-ce829800-950d-11ea-9178-2f78a46981d3.png)

#### c. Locations from where most number of tweets are posted

![image](https://user-images.githubusercontent.com/52653296/81777150-e1956800-950d-11ea-8b8f-58c37a4063da.png)

#### d. Locations from where most number of fake tweets are posted

![image](https://user-images.githubusercontent.com/52653296/81777189-efe38400-950d-11ea-98e4-bfa68e7eac45.png)

# Performance measure from different ML Algorithm 

![image](https://user-images.githubusercontent.com/52653296/81777312-2de0a800-950e-11ea-9818-3ce3b0b78f27.png)

After Performance Tuning, Logistic Regression model performance was increase to 78%.

![image](https://user-images.githubusercontent.com/52653296/81777340-3e911e00-950e-11ea-899b-921f5f75fd30.png)


## Blog Link : https://medium.com/@himanshubajpai869/real-or-not-nlp-with-disaster-tweets-77ab3ba8325f

# Acknowledgements :

1. The Dataset can be obtained from here : https://www.kaggle.com/c/nlp-getting-started/data

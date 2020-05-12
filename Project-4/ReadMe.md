Business Understanding :

Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster.

In this project , we will be building a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

Dataset is taken from Kaggle : https://www.kaggle.com/c/nlp-getting-started/data

Project Overview :

1. EDA on the Dataset 
2. Data Cleansing
3. Apply Different Machine Learning Models and evaluate Results.
4. Select the Best Performing Model and tune it to improve its performance
5. Conclusion
6. Future Improvements


Visualizations from the Notebook :

a. Target Distribution in Training Dataset

![image](https://user-images.githubusercontent.com/52653296/81724405-f63c1680-94a1-11ea-804b-a4156ee29850.png)

b. Frequent Hashtags in Training Set

![image](https://user-images.githubusercontent.com/52653296/81724679-677bc980-94a2-11ea-9017-02062514bb5e.png)

c. Locations from where most number of tweets are posted

![image](https://user-images.githubusercontent.com/52653296/81724777-9003c380-94a2-11ea-88b8-545c8cd6c42c.png)

d. Locations from where most number of fake tweets are posted

![image](https://user-images.githubusercontent.com/52653296/81724844-a7db4780-94a2-11ea-8be2-7f551220073a.png)

Performance measure from different ML Algorithm 

![image](https://user-images.githubusercontent.com/52653296/81724942-c9d4ca00-94a2-11ea-8fdc-ddd85c7d8f87.png)

After Performance Tuning, Logistic Regression model performance was increase to 69%.

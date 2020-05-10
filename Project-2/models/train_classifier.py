import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterDatasetClean',engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    return X,Y,category_names

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans(PUNCT_TO_REMOVE,' '*len(string.punctuation)))

def tokenize(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = lemmatize_words(text)
    return nltk.word_tokenize(text)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def build_model():
    pipeline = Pipeline(
    [
        ('vectorizer',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer(norm='l1')),
        ('classifer', MultiOutputClassifier(RandomForestClassifier(min_samples_split=2)))
    ])
    
    return pipeline

from sklearn.metrics import classification_report
def classification_report_for_each_column(y_test, y_pred):
    for i, col in enumerate(y_test.columns): 
        print('-------:',col,':-------')
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))
        
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    classification_report_for_each_column(Y_test, y_pred)

from sklearn.externals import joblib
def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
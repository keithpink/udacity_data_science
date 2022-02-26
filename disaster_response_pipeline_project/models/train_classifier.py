
"""
Classifier Trainer
Project: Disaster Response Pipeline 

Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl

Arguments:
    1) Path to SQLite destination database (e.g. DisasterResponse.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

import sys

# import NPL libraries
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['words','punkt','stopwords','wordnet', 'averaged_perceptron_tagger'])

# import ML libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    ''' load data from database
        return X, Y and category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT*FROM MessageCategories', engine)

    category_names = df.columns[4:]
    X = df['message'].values[:]
    Y = df[category_names].values[:]

    return X, Y, category_names


def tokenize(text):
    ''' given raw text
        return cleaned and tokenized text
    '''
    
    # normalize case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    clean_tokens = list()
    for tok in tokens:
        if tok not in stopwords.words("english"):
            clean_tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' Build a machine learning and Gridsearch pipeline
        return the model with params tuned
    '''
    
    # machine learning pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        
        ])),
        
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # grid search the hyperparameter
    parameters = {
        'clf__estimator__n_estimators': [20, 50],
        'clf__estimator__min_samples_split': [2, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Show the accuracy, precision, and recall of the tuned model 
        for each output category of the dataset
    '''
    Y_pred_tuned = model.predict(X_test)

    for i in range(len(category_names)):
        print(classification_report(Y_test[:,i], Y_pred_tuned[:,i], target_names=category_names[i]))
        accuracy = (Y_test[:,0]==Y_pred_tuned[:,0]).mean()
        print('Accuracy is {}'.format(accuracy))

    return


def save_model(model, model_filepath):
    ''' save model to the given filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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
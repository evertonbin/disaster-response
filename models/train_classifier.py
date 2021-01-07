import sys
sys.path.append("../models")
from customized_transformers import TextTokenizer, ModalVerbCounter, NumeralCounter, ElectricityWordCounter, CharacterCounter, CapitalLetterCounter, StartsWithFirstPersonPron
import pandas as pd
import numpy as np
import nltk
import re
import pickle

from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    '''It accesses SQLite database and loads data from DisasterResponseTable.

    Input:
    database_filepath: path for accessing SQLite database

    Outputs:
    X: array containing the messages used for train and test sets
    y: array containg the labels assigned for each message
    category_names: list containing the name of the different classes
    '''
    # Creating SQLite engine to access database:
    engine = create_engine('sqlite:///'+database_filepath)
    # Reading the DisasterResponseTable in the database:
    df = pd.read_sql_table('DisasterResponseTable', engine)
    # Selecting messages and labels arrays as X and y:
    X = df.message.values
    y = df.iloc[:,4:].values
    # Extracting category names:
    category_names = list(df.columns)[4:]

    return X, y, category_names

def build_model():
    '''It creates pipeline for text transformations, and applies the pipeline
    to a GridSearchCV algorithm over specific parameters for model tuning.

    Output:
    cv: GridSearchCV object instatiated over pipeline sequence with a
    MultinomialNB classifier
    '''
    # Creating pipeline with feature union and MultinominalNB classifier:
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('tknz', TextTokenizer()),
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])),
            ('md_count', ModalVerbCounter()),
            ('cd_count', NumeralCounter()),
            ('elec_count', ElectricityWordCounter()),
            ('txt_length', CharacterCounter()),
            ('cap_length', CapitalLetterCounter()),
            ('first_pron', StartsWithFirstPersonPron())
        ])),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])
    # Setting parameters for GridSearchCV:
    parameters = {
        'features__nlp_pipeline__vect__ngram_range': [(1,3), (1,4)],
        'features__nlp_pipeline__vect__min_df': [2, 3],
        'features__nlp_pipeline__tfidf__norm': ['l1', 'l2'],
        'clf__estimator__alpha': [0.1, 0.3]
    }
    # Instatiating GridSearchCV:
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    '''It recieves the model fitted over train data, makes prediction over
    the test set and evaluates the model for each category.

    Inputs:
    model: MultinomialNB model fitted over train set
    X_test: array with messages for test set
    y_test: array labeling messages in the test sets
    category_names: list with the corresponding category name
    '''
    # Making predictions over test set:
    y_pred = model.predict(X_test)
    # Printing classification_report for each category over test set:
    for true, pred, cat in zip(y_test, y_pred, category_names):
        print(cat)
        print(classification_report(true, pred))


def save_model(model, model_filepath):
    '''It saves the model in a pickle file to be used in production

    Inputs:
    model: MultinomialNB model trained and evaluated in previous steps
    model_filepath: file path and file name to save the model
    '''
    # Saving model as a pickle file:
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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

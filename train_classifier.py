import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Defining caracter counter estimator to create new feature:
class CharacterCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        '''It recieves an array of messages and counts the number of characters
        for each message.

        Input:
        X: array of text messages

        Output:
        n_caract: array with number of caracteres for each message
        '''
        # Creating empty list:
        n_caract = list()
        # Counting caracteres:
        for text in X:
            n_caract.append(len(text))
        n_caract = np.array(n_caract)
        print('##############################################################')
        print(n_caract, n_caract.shape, len(n_caract))
        n_caract = n_caract.reshape((len(n_caract),1))
        return n_caract

# Defining capital letter counter estimator to create new feature:
class CapitalLetterCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        '''It recieves an array of messages and counts the number of capital
        letters for each message.

        Input:
        X: array of text messages

        Output:
        cap_count_arr: array with the number of capital letters for each message
        '''
        # Creating empty list:
        cap_count_list = list()
        # Verifying each character to see whether it's a capital letter or not:
        for i in range (len(X)):
            cap_count = 0
            msg = X[i]
            for j in range(len(msg)):
                if msg[j].isupper():
                    cap_count += 1
            cap_count_list.append(cap_count)
        # Transforming list into array:
        cap_count_arr = np.array(cap_count_list)

        print('##############################################################')
        print(cap_count_arr, cap_count_arr.shape, len(cap_count_arr))
        cap_count_arr = cap_count_arr.reshape((len(cap_count_arr),1))
        return cap_count_arr

# Defining first person pronoun estimator to create new feature:
class StartsWithFirstPersonPron(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        '''It recieves an array of messages and evaluates whether each message
        starts with one of the first person pronouns (I, my, we, our).

        Input:
        X: array of text messages

        Output:
        first_person_arr: array indicating whether each message starts with
        first person pronoun
        '''
        # Creating empty list:
        first_person = list()
        # Creating list for target first person pronouns:
        pron_list = ['i', 'my', 'we', 'our']
        # Tokenizing message and verifying whether first token is a first person
        # pronoun in the list or not:
        for text in X:
            tokens = word_tokenize(text.lower())
            if len(tokens) > 0:
                if tokens[0] in pron_list:
                    first_person.append(1)
                else:
                    first_person.append(0)
            else:
                first_person.append(0)
        # Transforming list into array:
        first_person_arr = np.array(first_person)

        print('##############################################################')
        print(first_person_arr, first_person_arr.shape, len(first_person_arr))
        first_person_arr = first_person_arr.reshape((len(first_person_arr),1))
        return first_person_arr

class Debug(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        print(X.shape)
        return X

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

def tokenize(text):
    '''It replaces all urls for a common placeholder, tokenizes and lemmatizes
    text, transforms text to lowercase, and removes blank spaces.

    Input:
    text: text message ('X' from load_data() function)

    Output:
    clean_tokens: list of tokens from text after transformation process
    '''
    # Getting list of all urls using regex:
    detected_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    # Replacing each url in text string with placeholder:
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    # Extracting tokens from text:
    tokens = word_tokenize(text)
    # Instantiating Lemmatizer object:
    lemmatizer = WordNetLemmatizer()
    # Applying transformations:
    clean_tokens = []
    for tok in tokens:
        # Lemmatizing, trnasforming to lowercase, and removing blank spaces:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # Adding transformed token to clean_tokens list:
        clean_tokens.append(clean_tok)

    return clean_tokens


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
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('debug1', Debug()),
                ('tfidf', TfidfTransformer()),
                ('debug2', Debug())
            ])),
            ('txt_length', CharacterCounter()),
            ('debug3', Debug()),
            ('cap_length', CapitalLetterCounter()),
            ('debug4', Debug()),
            ('first_pron', StartsWithFirstPersonPron()),
            ('debug5', Debug())
        ])),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])
    # Setting parameters for GridSearchCV:
    parameters = {
        'features__nlp_pipeline__vect__stop_words': ['english'],
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

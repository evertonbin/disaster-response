import sys
sys.path.append("/models")
from customized_transformers import TextTokenizer, ModalVerbCounter, NumeralCounter, ElectricityWordCounter, CharacterCounter, CapitalLetterCounter, StartsWithFirstPersonPron

import json
import plotly
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM DisasterResponseTable', engine)
# load model
model = joblib.load("models/nb_classifier.pkl")
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Extracting data needed for visuals
    # Mmessage count by type (genre):
    genre_msg_count = df.groupby('genre').count()['message']
    genre = list(genre_msg_count.index)

    # Finding the top 10 categories with the highest and lowest % of messages and display:
    prop_cats_df = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)#.sort_values(ascending = False)[:,0:10]
    # Highest % messages:
    top_cats_df = (prop_cats_df.sort_values(ascending=False))
    top_cats_names = list(top_cats_df[0:10].index)
    top_cats_proportions = top_cats_df[0:10]
    # Lowest % messages:
    lowest_cats_names = list(top_cats_df[-11:].index)
    lowest_cats_proportions = top_cats_df[-11:]

    # Finding top tfidf words by genre
    text_tokenizer = TextTokenizer()

    # Direct:
    genre_direct = df[df['genre'] == 'direct']
    genre_direct = genre_direct.message.values
    genre_direct_tok = text_tokenizer.transform(genre_direct)
    tfidf_direct = TfidfVectorizer(min_df = 3).fit(genre_direct_tok)
    feature_names_direct = np.array(tfidf_direct.get_feature_names())
    # sum tfidf frequency of each term through documents
    sums_direct = tfidf_direct.transform(genre_direct_tok).sum(axis=0)
    # connecting term to its sums frequency
    data_direct = []
    for col, term in enumerate(feature_names_direct):
        data_direct.append( (term, sums_direct[0,col] ))
    ranking_direct = pd.DataFrame(data_direct, columns=['term','rank']).sort_values('rank', ascending=False)
    terms_direct = list()
    tf_direct = list()
    for i in range(10):
        terms_direct.append(ranking_direct['term'].iloc[i])
        tf_direct.append(ranking_direct['rank'].iloc[i])

    # News:
    genre_news = df[df['genre'] == 'news']
    genre_news = genre_news.message.values
    genre_news_tok = text_tokenizer.transform(genre_news)
    tfidf_news = TfidfVectorizer(min_df = 3).fit(genre_news_tok)
    feature_names_news = np.array(tfidf_news.get_feature_names())
    # sum tfidf frequency of each term through documents
    sums_news = tfidf_news.transform(genre_news_tok).sum(axis=0)
    # connecting term to its sums frequency
    data_news = []
    for col, term in enumerate(feature_names_news):
        data_news.append( (term, sums_news[0,col] ))
    ranking_news = pd.DataFrame(data_news, columns=['term','rank']).sort_values('rank', ascending=False)
    terms_news = list()
    tf_news = list()
    for i in range(10):
        terms_news.append(ranking_news['term'].iloc[i])
        tf_news.append(ranking_news['rank'].iloc[i])

    # Social:
    genre_social = df[df['genre'] == 'social']
    genre_social = genre_social.message.values
    genre_social_tok = text_tokenizer.transform(genre_social)
    tfidf_social = TfidfVectorizer(min_df = 3).fit(genre_social_tok)
    feature_names_social = np.array(tfidf_social.get_feature_names())
    # sum tfidf frequency of each term through documents
    sums_social = tfidf_social.transform(genre_social_tok).sum(axis=0)
    # connecting term to its sums frequency
    data_social = []
    for col, term in enumerate(feature_names_social):
        data_social.append( (term, sums_social[0,col] ))
    ranking_social = pd.DataFrame(data_social, columns=['term','rank']).sort_values('rank', ascending=False)
    terms_social = list()
    tf_social = list()
    for i in range(10):
        terms_social.append(ranking_social['term'].iloc[i])
        tf_social.append(ranking_social['rank'].iloc[i])

    # create visuals
    graphs = [
        # Bar graph for messages count by Genre
        {
            'data': [
                    Bar(x = genre,
                        y = genre_msg_count,
                        name = 'Message Counts')
                    ],
            'layout': {
                'title': 'Distribution of Messages by Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        },
        # Bar graph with top 10 categories with highest % of messages:
        {
            'data': [
                Bar(
                    x = top_cats_names,
                    y = top_cats_proportions
                )
            ],

            'layout': {
                'title': 'Top 10 Categories by Proportion of Messages Received',
                'yaxis': {
                    'title': "Proportions"
                },
                'xaxis': {
                    'title': "Message Types"
                }}},
        # Bar graph with 10 categories with lowest % of messages:
        {
            'data': [
                Bar(
                    x = lowest_cats_names,
                    y = lowest_cats_proportions
                )
            ],

            'layout': {
                'title': '10 Categories with Lowest Proportion of Messages Received',
                'yaxis': {
                    'title': "Proportions"
                },
                'xaxis': {
                    'title': "Message Types"
                }}},
        # Bar graph with top 10 tfidf for direct messages:
        {
            'data': [
                Bar(
                    x = terms_direct,
                    y = tf_direct,
                    textposition = 'auto'
                )
            ],

            'layout': {
                'title': '10 Words with Highest Term-Frequency for Direct Messages',
                'yaxis': {
                    'title': "Tf-idf"
                },
                'xaxis': {
                    'title': "Words"
                }}},
        # Bar graph with top 10 tfidf for news messages:
        {
            'data': [
                Bar(
                    x = terms_news,
                    y = tf_news,
                    textposition = 'auto'
                )
            ],

            'layout': {
                'title': '10 Words with Highest Term-Frequency for News Messages',
                'yaxis': {
                    'title': "Tf-idf"
                },
                'xaxis': {
                    'title': "Words"
                }}},
        # Bar graph with top 10 tfidf for social media messages:
        {
            'data': [
                Bar(
                    x = terms_social,
                    y = tf_social,
                    textposition = 'auto'
                )
            ],

            'layout': {
                'title': '10 Words with Highest Term-Frequency for Social Media Messages',
                'yaxis': {
                    'title': "Tf-idf"
                },
                'xaxis': {
                    'title': "Words"
                }}}]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
#def main():
#    app.run(host='0.0.0.0', port=3001, debug=True)
#if __name__ == '__main__':
#    main()

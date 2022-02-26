import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #genre and aid_related status
    aid_rel1 = df[df['aid_related']==1].groupby('genre').count()['message']
    aid_rel0 = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_names = list(aid_rel1.index)

    
    new_df = df.drop(['id', 'message', 'original'], axis = 1)
    
    def dist(genre): 
    ''' get the distribution of the given genre
    '''
        genre_dist = new_df[new_df['genre'] == genre].drop('genre', axis=1)
        genre_1 = genre_dist.sum()/len(genre_dist)
        genre_1 = genre_1.sort_values(ascending=False)
        genre_0 = 1 - (genre_1)
        genre_name = genre_1.index
        return genre_1, genre_0, genre_name
    
    # apply the function for each genre
    direct_1, direct_0, direct_name = dist('direct')
    news_1, news_0, news_name = dist('news')
    social_1, social_0, social_name = dist('social')
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_rel1,
                    name = 'Aid related'

                ),
                Bar(
                    x=genre_names,
                    y= aid_rel0,
                    name = 'Aid not related'
                )
            ],

            'layout': {
                'title': 'Distribution of message by genre and \'aid related\' class ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
        
        # plot distribution of Direct
        {
            'data': [
                Bar(
                    x=direct_name,
                    y=direct_1,
                    name = 'Class = 1'
                ),
                Bar(
                    x=direct_name,
                    y=direct_0,
                    name = 'Class = 0',
                    marker = dict(
                            color = 'rgb(181, 205, 235)'
                                )
                )
            ],

            'layout': {
                'title': 'Distribution of labels: Direct',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "",
            #        'tickangle': -45
                },
                'barmode' : 'stack'
            }
        },
        
        # plot distribution news
        {
            'data': [
                Bar(
                    x=news_name,
                    y=news_1,
                    name = 'Class = 1'
                ),
                Bar(
                    x=news_name,
                    y=news_0,
                    name = 'Class = 0',
                    marker = dict(
                            color = 'rgb(181, 205, 235)'
                                )
                )
            ],

            'layout': {
                'title': 'Distribution of labels: News',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "",
            #        'tickangle': -45
                },
                'barmode' : 'stack'
            }
        },
        
        # plot distribution of social
        {
            'data': [
                Bar(
                    x=social_name,
                    y=social_1,
                    name = 'Class = 1'
                ),
                Bar(
                    x=social_name,
                    y=social_0,
                    name = 'Class = 0',
                    marker = dict(
                            color = 'rgb(181, 205, 235)'
                                )
                )
            ],

            'layout': {
                'title': 'Distribution of labels: Social',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "",
            #        'tickangle': -45
                },
                'barmode' : 'stack'
            }
        }
    ]
    
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


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
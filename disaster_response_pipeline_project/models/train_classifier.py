import sys


def load_data(database_filepath):
    ''' load data from database
        return X, Y and category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT*FROM MessageCategories', engine)

    category_names = df.columns[4:]
    X = df['message'].values[:50]
    Y = df[category_names].values[:50]

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
        take in message column as input and output classification results on the other 36 categories 
    '''
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Report the f1 score, precision and recall 
        for each output category of the dataset
    '''
    pass


def save_model(model, model_filepath):
    pass


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
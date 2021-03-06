"""
Preprocessing of Data
Project: Disaster Response Pipeline 

Sample Script Syntax:
> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>

Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db

Arguments Description:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. DisasterResponse.db)
"""


import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' load two datasets and merge them into one dataframe
        return the merged dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, how='outer', on='id')
    return df


def clean_data(df):
    ''' clean the given dataframe
        return the cleaned dataframe
    '''

    # Split categories into separate category columns
    # use this row to extract a list of new column names for categories.
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[-1] )
        categories[column] = categories[column].apply(lambda x: int(x))


    # Replace categories column in df with new category columns.
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop column 'child_alone' which only contains 0
    df.drop('child_alone', axis=1, inplace=True)
    # drop rows realted = 2
    df = df.loc[df['related'] != 2]

    return df


def save_data(df, database_filepath):
    ''' Save the clean dataset into an sqlite database.
    '''
    engine = create_engine('sqlite:///'+str(database_filepath))
    df.to_sql('MessageCategories', engine, index=False, if_exists = 'replace')


def main():
    ''' ETL pipeline process
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.shape)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
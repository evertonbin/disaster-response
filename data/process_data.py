import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Loads the .csv files (messages and categories), and merge them together
    into one dataframe.

    Inputs:
    msg_file_path: path for the messages.csv file
    ctg_file_path: path for the categories.csv file

    Output:
    df: concatenated messages and categories dataframe
    '''
    # Loading messages.csv file:
    messages = pd.read_csv(messages_filepath)
    # Loading categories.csv file:
    categories = pd.read_csv(categories_filepath)
    # Merging messages and categories dataframes together:
    df = messages.merge(categories, on = 'id')

    return df


def clean_data(df):
    '''Recieves messages and categories merged dataframe and applies cleansing
    process: splits categories into different columns, change its values to 0|1
    format and drops duplicate rows.
    Input:
    df: dataframe returned from the load_data() function

    Output:
    df: dataframe after cleansing and transformations
    '''
    # Expanding different categories to different columns:
    categories = df.categories.str.split(';', expand = True)
    # Using first row to rename columns:
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    # Renaming categories column names:
    categories.columns = category_colnames
    # Altering categories values to 0|1 format:
    for column in categories:
        # setting each value to its last character:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # converting column from string to numeric:
        categories[column] = pd.to_numeric(categories[column])
    # Dropping the original 'categories' column from the dataframe:
    df.drop('categories', axis = 1, inplace = True)
    # Concatenating dataframe with the new categories dataframe:
    df = df.merge(categories, right_index = True, left_index = True)
    # Dropping duplicates:
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, db_filename):
    '''Creates SQLite database and table for the data from the dataframe after
    cleansing process.
    Input:
    df: dataframe returned from the clean_data() function
    db_filename: name of the SQLite database
    '''
    # Creating SQLite database (engine):
    engine = create_engine('sqlite:///'+db_filename)
    # Creating table in the database with the data from 'df' dataframe
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists = 'replace')

    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

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
# Disaster Response Pipeline Project

![overview1](https://github.com/keithpink/udacity_data_science/blob/main/disaster_response_pipeline_project/sceenshot/overview.png)

## Table of Contents

1. [Project Overview](#overview)
2. [File Descriptions](#files)
3. [Installation](#installation)
4. [License](#license)

## 1. Project Overview <a name="overview"></a>

The purpose of the project is to analyze disaster data from [Appen](https://appen.com/) (formally Figure 8) to build a model for an API that classifies disaster messages.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.




## 2. File Descriptions <a name="files"></a>
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- Screenshots
          |-- README
~~~~~~~

1. App folder including the templates folder and "run.py" for the web application
2. Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
4. README file

## 3. Installation<a name="installation"></a>


### Installation:

This project requires Python 3.x and the following Python libraries installed:
Flask, Json, Matplotlib, Nltk, NumPy, Pandas, Pickle, Plotly, Re, Sklearn, Sqlalchemy, Sys.



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/. Input a new message and get classification results in several categories

![overview2](https://github.com/keithpink/udacity_data_science/blob/main/disaster_response_pipeline_project/sceenshot/overview2.png)

## 4. License<a name="license"></a>
* This app was completed as part of the Udacity Data Scientist Nanodegree. 
* Code templates and data were provided by Udacity. 
* The data was originally sourced by Udacity from Figure Eight.

# Disaster Response Pipeline Project

### Table of Contents

1. [Instructions](#instructions)
2. [Installation](#installation)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model:
        `python models/train_classifier.py data/DisasterResponse.db models/nb_classifier.pkl`

2. Uncomment lines 235 to 238 in the run.py (app's directory) file and run the following command in the app's directory to run the web app on your local computer:
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/

## Installation <a name="installation"></a>

Most of the necessary libraries used in this project are already available in Anaconda distribution of Python.
Libraries that need previous installation:
1. [NLTK](https://www.nltk.org/)
2. [SQLalchemy](https://www.sqlalchemy.org/)
3. [Json](https://docs.python.org/3/library/json.html)
4. [Plotly](https://plotly.com/)
5. [Flask](https://flask.palletsprojects.com/en/1.1.x/)

This script was written using Python version 3.*.

## Project Motivation<a name="motivation"></a>

Whenever a disaster happens, whether it's a storm or an earthquake, people tend to communicate to get help. This communication can go from direct messages to social media posts.

The point is that, in circumstances like that, the number of messages can be huge, making it difficult to classify each one of them and direct them to the right authorities that would be responsible for the different claims that could go from an electricity issue to roads that got blocked after a storm.

This project is built under a dataset provided by [FigureEight](https://appen.com/), with prelabeled tweets and text messages from real-life disasters. It executes an ETL pipeline for the data and a Machine Learning pipeline that trains a supervised model to automatically classify messages in 36 different classes, including weather_related, storm, earthquake, food, water, electricity, and so on.

The challenge is to tune the model in a way that it can capture all the main topics in one text, given the fact that one message can belong to several different classes. Besides that, there some classes that have few examples, making it difficult for the model to precisely 'understand' the subject.

In real life, this could be an important tool to allow that these messages would actually be delivered to the right departments, providing a faster response while helping the ones in need.

## File Descriptions <a name="files"></a>

1. disaster_messages.csv: .csv file containing the raw data with the disaster messages.
2. disaster_categories.csv: .csv file with the labels for each message.
3. process_data.py: ETL pipeline for reading the .csv files, cleaning data and storing in a database - DisasterResponse.db.
4. customized_transformers.py: it defines customized transformers to be applied during machine learning pipeline process.
5. train_classifier.py: machine learning pipeline that reads the stored data and performs a GridSearchCV for training the message classifier.
6. run.py: it uses the data to create Plotly visualizations to be presented in the web app. It also renders the web app in the local machine.
7. DisasterResponse.db: database containing the data after being processed by the ETL pipeline.
8. nb_classifier.pkl: Naïve Bayes classifier trained over the machine learning pipeline.
9. master.html: Bootstrap index webpage of the web app, containing visualizations and the form for typing the message to be classified.
10. go.html: Bootstrap webpage that presents the labels related for the message, according to the nb_classifier.
11. Procfile: this files serves for deployment purposes, indicating to Heroku what to do when starting the web app.
12. requirements.txt: it lists all the libraries the web app relies on (also for deployment purposes).

## Results<a name="results"></a>

This project results in a web app that can be deployed or run locally and used to classify a disaster message according to its related topics.

It's important to say that further improvements could be developed, especially for increasing the recall for labels that have few or none observations in the dataset, or even for labels that can't be noticed for the model, given the context.

The model was build with the purpose of prioritizing the Recall metric, given that, in real life, it would be better to have false positives than false negatives, once that false negatives would lead to disaster messages that would not be properly delivered.

It was achieved an average recall close to 75% over the test set, but some classes like 'water' are still tricky for the model to understand.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to the [FigureEight](https://appen.com/) company for providing the prelabeled data, and to [Udacity](https://www.udacity.com/) for proposing this amazing project that results in a direct impact on people's lives.

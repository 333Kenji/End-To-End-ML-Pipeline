import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

## Modelling
from sklearn import svm
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import log_loss, make_scorer, confusion_matrix, plot_confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

## Flask

from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, url_for, session, redirect, jsonify, render_template_string, request

from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

# some items may not apply to current build.

## Integrating file as object
app = Flask(__name__)

## Following two functions plus the last are the primary components
#   of application besides data pipeline and processing.
# 1. Rendering html page for user input
# 2. Activating data processing function, passing in user input as parameter.
# 3. Calculates and returns GridSearch metrics.
#    Also activates and takes input from data pipeline, processing, and API call functions.

@app.route('/gridsearch')
def my_form():
    """[Renders initial app page if route navigated to.]

    Returns:
        [html]: [page with input field for user text input]
    """
    return render_template('form.html')

## 
# request and formats for use in API call.
@app.route('/gridsearch', methods=['POST'])
def my_form_post():
    """[Receives user text entry via request associated user text input field
        on initial app html page
        Also sets to upper.
        // Opportunity to further explore user input formatting and security]

    Returns:
        [function object]: [Calls a function that outputs (as rendered html)
            application output as html page.
        Said function requires input from several other functions, in sequence,
            all requiring user text inut string.]
    """    
    text = request.form['text']
    ticker = text.upper()
    return gridsearch(ticker)


@app.route('/gridsearch', methods=("POST", "GET"))
def index(ticker):
    """[Makes API calls using user input string. API provider returns market data for asset
        represented by user input string]

    Args:
        ticker ([string]): [User input as passed through to this initial and
            utilizationand primary utilization: to be passed into the API call.]

    Returns:
        [pandas Dataframe]: [Collated market data and technical indicator data]
    """    
    API_key = 'CSMN0LYTQ5UYMVUT'
    API_URL = "https://www.alphavantage.co/query"

    ts = TimeSeries(key=API_key, output_format='pandas')
    request = TechIndicators(key=API_key, output_format='pandas')
    
    price = ts.get_daily_adjusted(ticker, outputsize='full')[0].transpose()
    price.rename(index={'1. open':'Open', '2. high':'High', '3. low':'Low', '4. close':'Close', '5. adjusted close':'Adjusted Close', '6. volume':'Volume', '7. dividend amount':'Dividend Amount', '8. split coefficient':'Split Coefficient'}, inplace=True)
    price = price.transpose()
    macd, meta_data = request.get_macd(symbol=ticker,interval='daily')
    bbands, meta_data = request.get_bbands(symbol=ticker,interval='daily')
    rsi,meta_data = request.get_rsi(symbol=ticker,interval='daily',time_period=15)
    sma5, meta_data = request.get_sma(symbol=ticker,time_period=5)
    sma15, meta_data = request.get_sma(symbol=ticker,time_period=15)
    roc, meta_data = request.get_roc(symbol=ticker,series_type='high',interval='daily')
    
        
    tables = pd.concat([price, macd, bbands, rsi, sma5, sma15, roc], axis=1)
    # tables = tables.set_index(pd.to_datetime(tables['date'].values)).dropna()
    # useful scrap
    return tables


# blocked out returns to 'render_template(..' were part of intial EDA functions.
# Retaining in several particular functions for possible data dislay usage.

@app.route('/', methods=('POST', 'GET'))
def ti_sig_feats(ticker):
    """[feature engineering]

    Args:
        ticker ([string]): [user input string, primarily present to be
        passed into API call function for use, and also as remnant of
        former indexing of 'db' which had tables for multiple tables.]

    Returns:
        [pandas DataFrame]: [Same table from above function but with new features for making
        explicit technical indicator feature signals and to create target]
    """    
    db = {}
    data = index(ticker)
    db[ticker]=data.dropna()

    d = 4
    later = db[ticker]['Adjusted Close'][d:]
    today = db[ticker]['Adjusted Close'][:-d]
    db[ticker]['change']=0
    db[ticker]['crossover']=0
    db[ticker]['thresh']=db[ticker]['Adjusted Close']*(1+(.1/365)*100)
    db[ticker]['target']=True
    # there's something interesting in thresh that I'll leave alone for now.
    # It is the value anticpated, whatever that may be.
    # One could make explicit the relation to the value that actually occurs or something for.
    db[ticker]['crossover'] = db[ticker]['MACD'] > db[ticker]['MACD_Signal']
    
    for i, j , idx in zip(later, today, db[ticker].index[d:]):
        db[ticker].loc[idx,'change']=(j-i)/i
        
    change = db[ticker]['change']
    db[ticker]=db[ticker][::-1]
    db[ticker]=db[ticker].dropna()
    db[ticker]=db[ticker][4:-4].copy()
      
    for k, idx in zip(change, db[ticker].index[:]):
        if k < 0:
            db[ticker].loc[idx,'target']=False
    
    

    # return render_template('index.html',  data=[db[ticker].to_html(classes='data')], titles=db[ticker].columns.values)
    return db

@app.route('/', methods=('POST', 'GET'))
def data_splitter(ticker):
    """[creates dict where keys are tickers and their values are 
        all their particular.containing all tables associated with
        a tic.
        Then creates new tables by indexing into the db object at
        hard coded dates. //    ]

    Args:
        ticker ([string]): [user input string]

    Returns:
        [tuple]: [test + train, train, test]
    """    
    
    #train_start = '2020-11-08'

    train_end = '2017-12-31'
    
    test_start = '2018-01-01'
    test_end = '2021-03-31'
    
    
    tables = {}
    master_train =[]
    master_test = []
    #holdout_start = '2021-02-10'
    #holdout_end = '2021-03-24'
    db = ti_sig_feats(ticker)

    for k, v in db.items():
        train = db[k].loc[:train_end,:].copy()
        test = db[k].loc[test_start:,].copy()
        # df_holdout = TSLA.loc[holdout_start:holdout_end,:].copy()
        
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        
        tables[k]=[v, train, test]
        master_train.append(train)
        master_test.append(test)

    all_test = pd.concat(master_train)
    all_train = pd.concat(master_test)
    return tables, train, test


@app.route('/', methods=('POST', 'GET'))
def X_y(ticker):
    """[Generates target array and features matrix for text and train sets]

    Args:
        ticker ([string]): [user input string]

    Returns:
        [tuple: [feature-target split for test and train]
    """    
    tables, train, test = data_splitter(ticker)

    X_train = train[['Open', 'High','Low','Adjusted Close','Volume','MACD','MACD_Hist','MACD_Signal','Real Middle Band','Real Upper Band','Real Lower Band','change','crossover','thresh', 'RSI', 'SMA', 'ROC']].copy()
    y_train = train['target'].values

    X_test = test[['Open', 'High','Low','Adjusted Close','Volume','MACD','MACD_Hist','MACD_Signal','Real Middle Band','Real Upper Band','Real Lower Band','change','crossover','thresh', 'RSI', 'SMA', 'ROC']].copy()
    y_test = test['target'].values

    # X_train = train[['Open', 'High','Low','Adjusted Close','Volume','MACD','MACD_Hist','MACD_Signal', 'change','crossover','thresh']].copy()
    # y_train = train['target'].values

    # X_test = test[['Open', 'High','Low','Adjusted Close','Volume','MACD','MACD_Hist','MACD_Signal', 'change','crossover','thresh']].copy()
    # y_test = test['target'].values
    return X_train, y_train, X_test, y_test

@app.route('/', methods=('POST', 'GET'))
def gridsearch(ticker):
    """[Iterates through dictionary of models:parameters, instantiating,
        fitting, and storing predictions and scores for each.]

    Args:
        ticker ([string]): [user input string passed through to 
            function call via html form request.]

    Returns:
        [html]: [page displaying GridSearchCV output in table format.]
    """    
    scores = {
        'Ticker':ticker
        }
    X_train, y_train, X_test, y_test = X_y(ticker)
    model_params = {
    'Random Forest': {
        'Model': RandomForestClassifier(),
        'Parameters': {
            'n_estimators': range(1,70),
            'criterion': ['gini','entropy'],
            'min_samples_split':[150, 220, 300], #not sure but for val curves
            'min_samples_leaf': [50, 150, 220],
            'max_features': ['sqrt'],

            
            }
        }
    }
    
    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp['Model'], mp['Parameters'], cv = 10, return_train_score=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict((X_test))
        scores['Model']=model_name
        scores['Best Score']=clf.best_score_
        scores['Best Parameters']=clf.best_params_
        scores['Best Estimator']=clf.best_estimator_
        scores['Score'] = f"Score = {clf.score(X_test, y_test)}"
        scores['Precision'] = f"Precision = {precision_score(y_test, y_pred, average='macro')}"
        scores['Recall'] = f"Recall = {recall_score(y_test, y_pred, average='macro')}"

    return render_template('index.html',  scores=scores)







if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
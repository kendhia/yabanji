from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
import os
import json
import tweets_by_user
import pandas as pd
from time import sleep
from scripts import tweets_ml
import shutil
import matplotlib as plt

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

app = Flask(__name__)
     
@app.route("/predict")
def result():

  account = request.args.get('account')
  if (account):
    df, tweets = tweets_by_user.create_data(account)
    df, tweets = pd.read_csv('scripts/test_dta.csv'), "scripts/test_dta.csv"
    prediction = tweets_ml.predect_from_df(tweets , "tweet")
    
    df["Classification"] = prediction[1:]
    q = len(df[df["Classification"] == 0])
    s = len(df[df["Classification"] == 4])
    series = pd.Series([s, q], index=["Positive Tweets", "Negative Tweets"], name="Tweet Insights 1")
    fig = series.plot(kind='pie',  figsize=(80, 80), fontsize=32, title="Tweet Insights 1").get_figure()
    fig.savefig('static/images/plot1.png')
    return render_template('result.html',  tables=[df.to_html(index=False)], titles=df.columns.values, name=account, tweets_num=q+s,
    pos_tweets = s, neg_tweets = q, algorithm = "Bayes Algorithm", url='/static/images/plot1.png')



if __name__ == "__main__":   
  app.run()
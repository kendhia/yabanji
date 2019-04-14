from flask import Flask, redirect, url_for, render_template, request
from werkzeug import secure_filename
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

@app.route("/")
def index():
  return render_template("index.html")
     

@app.route("/uploader", methods=['POST'])
def uploader():
  if request.method == 'POST':
    try:
      shutil.rmtree("/home/fykendhia/Documents/yabanji/social_mapper/Input-Examples/imagefolder")
      os.mkdir("/home/fykendhia/Documents/yabanji/social_mapper/Input-Examples/imagefolder")
      os.remove("/home/fykendhia/Documents/yabanji/social_mapper/SM-Results/results-social-mapper.csv")
      os.remove("/home/fykendhia/Documents/yabanji/social_mapper/static/images/plot1.png")
    except Exception as e:
      pass
    f = request.files['file']
    f.save(os.path.join("Input-Examples/imagefolder", secure_filename(f.filename)))

    return redirect(url_for("result"))


@app.route("/result")
def result():
  
  os.system('python social_mapper.py -f imagefolder -i ./Input-Examples/imagefolder/ -m accurate -tw')

  done = False
  while(not done):
    sleep(5)
    try: 
      with open("config.json", "r") as fp:
        done = json.load(fp)["done"]
    except Exception as e:
      pass


  df = pd.read_csv("/home/fykendhia/Documents/yabanji/social_mapper/SM-Results/results-social-mapper.csv", header=0, sep=",")

  if (df["Twitter"][0] and len(df["Twitter"][0]) > 1):
    url = df["Twitter"][0]
    if (url):
      df, tweets = tweets_by_user.create_data(url.split("/")[3])
      prediction = tweets_ml.predect_from_df(tweets , "tweet")
      
      df["Classification"] = prediction[1:]
      q = len(df[df["Classification"] == 0])
      s = len(df[df["Classification"] == 4])
      series = pd.Series([s, q], index=["Positive Tweets", "Negative Tweets"], name="Tweet Insights 1")
      fig = series.plot(kind='pie',  figsize=(80, 80), fontsize=32, title="Tweet Insights 1").get_figure()
      fig.savefig('/home/fykendhia/Documents/yabanji/social_mapper/static/images/plot1.png')
      return render_template('result.html',  tables=[df.to_html(index=False)], titles=df.columns.values, name=url.split("/")[3], tweets_num=q+s,
      pos_tweets = s, neg_tweets = q, algorithm = "Bayes Algorithm", url='/static/images/plot1.png')

  return


if __name__ == "__main__":   
  app.run()
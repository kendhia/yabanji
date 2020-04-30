from twitterscraper.query import query_tweets_from_user
import pandas as pd


def create_data(user_name):
    df = pd.DataFrame(columns=['tweet'])
    a = query_tweets_from_user(user_name, 100)
    for i, tweet in enumerate(a):
        df.loc[i] = tweet.text.replace("\n", ",")
    df.to_csv("scripts/test_dta.csv", index=False, encoding="utf-8")

    return df, "scripts/test_dta.csv"
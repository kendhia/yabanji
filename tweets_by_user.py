from twitterscraper.query import query_tweets_from_user
import pandas as pd


def create_data(user_name):


    df = pd.DataFrame(columns=['tweet'])
    a = query_tweets_from_user(user_name, 100)

    for i, tweet in enumerate(a):
        df.loc[i] = tweet.text.encode('utf-8').replace("\n", ",")
            
    
    df.to_csv("/home/fykendhia/Documents/yabanji/social_mapper/scripts/test_dta.csv", index=False, encoding="utf-8")

    

    return df, "/home/fykendhia/Documents/yabanji/social_mapper/scripts/test_dta.csv"
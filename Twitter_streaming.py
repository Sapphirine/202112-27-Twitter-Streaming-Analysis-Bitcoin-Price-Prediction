# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# import tweepy library for twitter api access and textblob libary for sentiment analysis
import csv
import tweepy
import numpy as np
from textblob import TextBlob
import datetime
import time


def main():
    # set twitter api credentials
    consumer_key = 'jqln3CzqBBltSzTKE3zpxN0ln'
    consumer_secret = 'PyElb7WZq8DDysPZmhTjXb6oqv5l7mPUNhFstpSGTGXd0Zod3d'
    access_token = '1435930311230111746-46yg9jKvIAS4sfesKxQnZ62j4zR7c2'
    access_token_secret = '5yZhPOS3TmC5dgLgGINZusIZ8xO7hysCI7GVC1WDlxaVW'
    
    # set path of csv file to save sentiment stats
    path = 'live_tweet.csv'
    f = open(path, "a")
    f1 = open('tweet_data', 'a')
    # access twitter api via tweepy methods
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    twitter_api = tweepy.API(auth)
    
    queries = ['bitcoin, price, crypto', 'bitcoin price', 'price crypto', 'elon musk crypto',
             'cryptocurrency', 'dogecoin', 'Ethereum ETH', 'Litecoin LTC', 'blockchain']
    
    while True:
        for Q in queries:
            # fetch tweets by keywords
            tweets = twitter_api.search_tweets(q=[Q], count=100)
            
            # get polarity
            polarity = get_polarity(tweets, f1)
            sentiment = np.mean(polarity)

            # save sentiment data to csv file
            f.write(str(sentiment))
            f.write(",")
        
        f.write(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
        print(datetime.datetime.now())
        f.write("\n")
        f.flush()
        time.sleep(600)


def get_polarity(tweets, f):
    # run polarity analysis on tweets
    
    tweet_polarity = []
    
    for tweet in tweets:
        f.write(tweet.text + '\n')
        analysis = TextBlob(tweet.text)
        tweet_polarity.append(analysis.sentiment.polarity)
    
    return tweet_polarity

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


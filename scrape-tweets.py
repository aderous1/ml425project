#!/usr/bin/env python3

import os
import json
import argparse
import csv

import tweepy
import pandas as pd

with open('creds.json', 'r') as f:
    creds = json.load(f)

auth = tweepy.OAuthHandler(creds['consumer_key'], creds['consumer_secret'])
auth.set_access_token(creds['access_key'], creds['access_secret'])
api = tweepy.API(auth)

def get_tweets(username):
    #set count to however many tweets you want
    number_of_tweets = 1000

    #get tweets
    tweets_for_csv = []
    for tweet in tweepy.Cursor(api.user_timeline, screen_name = username, tweet_mode='extended').items(number_of_tweets):
        #create array of tweet information: username, tweet id, date/time, text
        tweets_for_csv.append([username, tweet.id_str, tweet.created_at, tweet.full_text])

    #write to a new csv file from the array of tweets
    outfile = f"data/tweets/{username}.csv"
    print(f"writing to {outfile}")
    with open(outfile, 'w+') as file:
        writer = csv.writer(file, delimiter=',', lineterminator="\n")
        writer.writerows(tweets_for_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "username",
        type=str,
        help="Twitter handle"
    )
    args = parser.parse_args()

    get_tweets(args.username)

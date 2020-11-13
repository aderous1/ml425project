import re
from pathlib import Path
import csv
from random import shuffle

if re.search('virus', 'CHINA VIRUS', re.IGNORECASE):
    print(True)

wordbank = [
    'corona','coronavirus','covid','covid19','covid-19','virus',
    'pandemic','epidemic','lockdown','testing','cases',
    'hospitalization','related deaths','mask','cdc',
    'conspiracy','sick','hospital','health'
]
wordbank_casesens = ['WHO',]

original_files = Path('tweets_filter_covid')
output = Path('all_filtered_randosort.csv')

writer = csv.writer(output.open('w'), delimiter=',', lineterminator="\n")

all_tweets = []

for orig_file in original_files.iterdir():
    if not orig_file.is_file():
        print(f"{orig_file} is not normal file")
        continue
    print(f"Loading {orig_file}...")
    reader = csv.reader(orig_file.open('r'))
    n = 0
    last = None
    try:
        for tweet in reader:
            last = tweet
            n += 1
            all_tweets.append(tweet)
        print(f"{n} items from {orig_file}...")
    except Exception as e:
        print(f"last: {last}")
        raise

shuffle(all_tweets)

print(f"{len(all_tweets)} total")

for t in all_tweets:
    writer.writerow(t)

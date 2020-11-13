import re
from pathlib import Path
import csv

if re.search('virus', 'CHINA VIRUS', re.IGNORECASE):
    print(True)

wordbank = [
    'corona','coronavirus','covid','covid19','covid-19','virus',
    'pandemic','epidemic','lockdown','testing','cases',
    'hospitalization','related deaths','mask','cdc',
    'conspiracy','sick','hospital','health'
]
wordbank_casesens = ['WHO',]

original_files = Path('tweets_1000')
output_dir = Path('tweets_filter_covid')

for orig_file in original_files.iterdir():
    if not orig_file.is_file():
        print(f"{orig_file} is not normal file")
        continue
    print(f"Loading {orig_file}...")
    reader = csv.reader(orig_file.open('r'))
    writer = csv.writer((output_dir / orig_file.name).open('w'))
    n = 0
    last = None
    try:
        for tweet in reader:
            last = tweet
            if len(tweet) == 0:
                continue
            if any(word in tweet[3].lower() for word in wordbank):
                writer.writerow(tweet)
                n += 1
                continue
            if any(word in tweet[3] for word in wordbank_casesens):
                writer.writerow(tweet)
                n += 1
                continue
        print(f"{orig_file} had {n} matches")
    except Exception as e:
        print(f"last: {last}")
        raise

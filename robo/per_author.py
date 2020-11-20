
import pandas as pd

rated = pd.read_csv('data/robo_model_predictions.csv')

rated.drop(rated.columns.difference(['author','predicted reliability']), 1, inplace=True)

counts = rated.groupby(['author', 'predicted reliability']).size()

counts.to_csv('data/robo_per_author_reliable_count.csv')

for a in set(rated['author']):
    f = rated[rated['author'] == a]
    gb = f.groupby('predicted reliability')
    vc = f['predicted reliability'].value_counts()
    print(f"{a},{vc[0] / vc[1]},{vc[0]},{vc[1]}")

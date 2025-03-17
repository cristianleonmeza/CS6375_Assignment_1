import json
import os
from collections import Counter

datasets = ["training.json", "validation.json", "test.json"]    #list of dataset file names

def load_json(filename):                                        #function to load JSON data from the files
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def get_statistics(data):                                       #function to compute statistics for a dataset
    total_reviews = len(data)
    star_counts = Counter(review["stars"] for review in data)
    return total_reviews, dict(star_counts)

for dataset in datasets:                                        #process each dataset
    if os.path.exists(dataset):
        data = load_json(dataset)
        total_reviews, star_distribution = get_statistics(data)
        
        print(f"{dataset}:")
        print(f"Total reviews: {total_reviews}")
        print("Star distribution:")
        for stars, count in sorted(star_distribution.items()):
            print(f"  {stars} stars: {count}")
        print("-" * 40)
    else:
        print(f"File {dataset} not found.")
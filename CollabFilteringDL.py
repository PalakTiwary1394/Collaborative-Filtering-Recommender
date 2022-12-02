import pandas as pd
import numpy as np


ratings_df = pd.read_csv("C:/Users/palak/Downloads/MockData/ratings_Movie.csv")
#print(ratings_df.head())

training_data = ratings_df.sample(frac=0.8, random_state=25)
print(training_data.head(5))

training_data.to_csv("C:/Users/palak/Downloads/MockData/ratings_Movie_train.csv", sep=',', encoding='utf-8', index=False)

testing_data = ratings_df.drop(training_data.index)
testing_data.to_csv("C:/Users/palak/Downloads/MockData/ratings_Movie_test.csv", sep=',', encoding='utf-8', index=False)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")



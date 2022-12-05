import tez
import pandas as pd
import numpy as np
import torch
from sklearn import model_selection
from sklearn import metrics
import torch
import torch.nn as nn

class MovieDataset():
    def _init_(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def _len_(self):
        return len(self.users)

    def _getitem_(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating =  self.ratings[item]

        return {"user" : torch.tensor(user, dtype=torch.long),
                "movie" : torch.tensor(movie, dtype=torch.long),
                "rating" : torch.tensor(rating, dtype=torch.float),}

class RecSysModel(tez.Model):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 32)
        self.movie_embed = nn.Embedding(num_movies, 32)
        self.out = nn.Linear(64, 1)

    def monitor_metrics(self, output, rating):
        output = output.detach().cpu().numpy()
        rating = rating.detach().cpu().numpy()
        return {
            'rmse' : np.sqrt(metrics.mean_squared_error(rating, output))
        }


    def forward(self, users, movies, ratings=None):
        user_embeds = self.user_embed(users)
        movie_embeds = self.user_embed(movies)
        rating_embeds = self.user_embed(ratings)
        output = torch.cat([user_embeds,movie_embeds], dim=1)
        output = self.out(output)

        loss = nn.MSELoss()(output, ratings.view(-1, 1))
        calc_metrics = self.monitor_metrics(output, ratings.view(-1, 1))
        return output,loss,calc_metrics



def train():
    # df = pd.read_csv("C:/Users/palak/Downloads/MockData/ratings_Movie_train.csv")
    df = pd.read_csv("C:/Users/pviswana/PycharmProjects/pythonProject/CollaborativeFiltering/mockdata_input/ratings_Movie_train.csv")
    #ID, user, movie, ratings
    df_train , df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.rating.values)

    train_dataset = MovieDataset(user=df_train.userId.values, movie=df_train.movieId.values, ratings=df_train.rating.values)

    valid_dataset = MovieDataset(user=df_valid.userId.values, movie=df_valid.movieId.values, ratings=df_valid.rating.values)

if __name__ == "__main__":
    train()


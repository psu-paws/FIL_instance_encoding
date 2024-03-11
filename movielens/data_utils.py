import sys
import pickle
import os

import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, train, clk_thres=5, raw_path="/checkpoint/kwmaeng/movielens-20/", processed_path="./movielens/processed/"):
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        processed_file = f"{processed_path}/movielens_processed_{clk_thres}star.pkl"

        if os.path.exists(processed_file):
            print("Reading processed input..")
            with open(processed_file, 'rb') as handle:
                data = pickle.load(handle)
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_test = data["x_test"]
            y_test = data["y_test"]
            max_genre_len = data["max_genre_len"]
            max_uid = data["max_uid"]
            max_mid = data["max_mid"]
            max_gid = data["max_gid"]
        else:
            print("Reading raw input..")
            
            movie_features = raw_path + "/movies.csv"
            interactions = raw_path + "/ratings.csv"

            movie_genres = {} # Key: movie_id, Val: list(movie cate)
            movie_id_remap = {}
            genre_id_remap = {}
            max_genre_len = 0
            if os.path.exists(movie_features):
                print(f"Reading movie features from {movie_features}")
                with open(movie_features) as f:
                    for j, line in enumerate(f):
                        if j == 0:
                            continue
                        parsed = line.split(",")
                        movie_id = int(parsed[0])
                        genres = parsed[-1].split("|")
                        genres_new = []

                        if movie_id not in movie_id_remap:
                            movie_id_remap[movie_id] = len(movie_id_remap) + 1 # 0 is reserved

                        for genre in genres:
                            if genre not in genre_id_remap:
                                genre_id_remap[genre] = len(genre_id_remap) + 1 # 0 is reserved for unknown
                            genres_new.append(genre_id_remap[genre])
                        max_genre_len = max(max_genre_len, len(genres_new))
                        assert(movie_id_remap[movie_id] not in movie_genres)
                        movie_genres[movie_id_remap[movie_id]] = genres_new + []

                print(f"Read {j} movies. Without duplicate, {len(movie_genres)}, max genre_len {max_genre_len}")
            else:
                sys.exit(f"ERROR: {movie_features} not found")

            user_logs = {} # Key: uid, val: List[(timestamp, features)]

            x_train = []
            y_train = []
            x_test = []
            y_test = []
            if os.path.exists(interactions):
                print(f"Reading interactions from {interactions}")
                # Read through the samples to fill user logs
                with open(interactions) as f:
                    for j, line in enumerate(f):
                        if j == 0:
                            continue
                        parsed = line.strip().split(",")
                        user_id = int(parsed[0])
                        movie_id = movie_id_remap[int(parsed[1])]
                        rating = float(parsed[2])
                        time = int(parsed[3])

                        clk = 1 if rating >= clk_thres else 0

                        assert(movie_id in movie_genres)

                        if user_id not in user_logs:
                            user_logs[user_id] = []
                        user_logs[user_id].append((time, clk, movie_id, movie_genres[movie_id] + []))
                    
                    for j, uid in enumerate(user_logs):
                        for i, (time, clk, movie_id, genre) in enumerate(sorted(user_logs[uid])):
                            # Train vs. Test split: 9:1
                            if i < len(user_logs[uid]) * 0.9:
                                y_train.append(clk)
                                x_train.append((uid, movie_id, genre + []))
                            else:
                                y_test.append(clk)
                                x_test.append((uid, movie_id, genre + []))

                    print(f"Read {j} user logs")

            max_uid = len(user_logs)
            max_mid = len(movie_id_remap)
            max_gid = len(genre_id_remap)

            data = {}
            data["x_train"] = x_train
            data["y_train"] = y_train
            data["x_test"] = x_test
            data["y_test"] = y_test
            data["max_genre_len"] = max_genre_len
            data["max_uid"] = max_uid
            data["max_mid"] = max_mid
            data["max_gid"] = max_gid

            with open(processed_file, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Saved")

        self.x = x_train if train else x_test 
        self.y = y_train if train else y_test 
        self.max_genre_len = max_genre_len
        self.max_uid = max_uid
        self.max_mid = max_mid
        self.max_gid = max_gid

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        #return ((torch.tensor(self.x[idx][0]), torch.tensor(self.x[idx][1]), torch.tensor(self.x[idx][2] + [0] * (self.max_genre_len - len(self.x[idx][2])))), torch.tensor(self.y[idx]))
        # Currently only using uid and mid
        return ((torch.tensor(self.x[idx][0]), torch.tensor(self.x[idx][1])), torch.tensor(self.y[idx]).reshape([1]).float())

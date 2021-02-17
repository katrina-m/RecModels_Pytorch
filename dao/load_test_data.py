import pandas as pd
import os
import numpy as np


def load_data(data_name):
    if data_name == "ml-1m":
        data_dir = "/tf/data/chenjiazhen/movie_data/movielens_data/ml_1m/"
        rating_df = pd.read_csv(os.path.join(data_dir, "processed_rating.csv"))
        #offset = rating_df.userId.max()
        #rating_df["itemId"] = rating_df.itemId + offset
        #rating_df.rename(columns={"userId":"srcId", "itemId":"dstId"}, inplace=True)
        #rating_df["eType"] = 0
        return rating_df
    else:
        pass


def load_movie_data(corpus_name, kg=False):
    if corpus_name == "ml-1m":
        data_dir = "/tf/data/chenjiazhen/movie_data/movielens_data/"
        rating_df = pd.read_csv(os.path.join(data_dir, "ml_1m", "ratings.csv"))
        if kg is True:
            kg_df = pd.read_csv(os.path.join(data_dir, "kg.csv"))
            mapping_df = pd.read_csv(os.path.join(data_dir, "mapping.csv")).astype(np.int32)
            item_id_map = dict(zip(mapping_df.itemId.values, mapping_df.entityId.values))
            # some itemId is not included in the knowledge graph, will append the id to the last.
            offset = max(kg_df.h.max(), kg_df.t.max())
            itemIds = []
            i=1
            for itemId in rating_df.itemId:
                if itemId in item_id_map:
                    itemIds.append(item_id_map[itemId])
                else:
                    itemIds.append(offset+i)
                    item_id_map[itemId] = offset+i
                    i += 1
            rating_df["itemId"] = itemIds
            return rating_df, kg_df
        else:
            return rating_df, None

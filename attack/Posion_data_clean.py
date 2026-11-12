import pandas as pd
import numpy as np

def posion_data_clean(attack_file, num_items):
    df = pd.read_csv(attack_file, sep=' ', header=None, names=['user', 'item', 'rating'])
    all_users = df['user'].unique()
    all_items = range(0, num_items)
    rating_matrix = pd.DataFrame(
        data=0.0,
        index=all_users,
        columns=all_items,
        dtype=np.float32
    )
    for user_id, item_id, rating in df.itertuples(index=False, name=None):
        rating_matrix.loc[user_id, item_id - 1] = float(rating)
    dense = rating_matrix.to_numpy(dtype=np.float32)
    dense = np.hstack([dense, np.ones((dense.shape[0], 1), dtype=np.float32)])
    return dense

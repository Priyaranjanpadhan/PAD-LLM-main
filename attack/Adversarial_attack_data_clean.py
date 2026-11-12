import numpy as np
from scipy.sparse import csr_matrix

def dataload(attack_file):
    data = np.load(attack_file)
    csr_matrix_data = csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape=data['shape']
    )
    dense_matrix = csr_matrix_data.toarray()
    ones_column = np.ones((dense_matrix.shape[0], 1))
    dense_matrix = np.hstack([dense_matrix, ones_column])
    return dense_matrix

def prepare_data_with_attack(normal_file, attack_file):
    user_item_pairs = np.loadtxt(normal_file, dtype=int, delimiter=',')
    num_users = np.max(user_item_pairs[:, 0]) + 1
    num_items = 3952
    rating_matrix = np.zeros((num_users, num_items))
    for uid, sid in user_item_pairs:
        rating_matrix[uid, sid] = 1
    ones_column = np.zeros((rating_matrix.shape[0], 1))
    rating_matrix = np.hstack([rating_matrix, ones_column])
    attack_matrix = dataload(attack_file)
    data_with_attack = np.vstack([rating_matrix, attack_matrix])
    return data_with_attack
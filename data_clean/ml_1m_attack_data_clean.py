from sklearn.model_selection import train_test_split
from attack.Adversarial_attack_data_clean import dataload
from attack.Posion_data_clean import posion_data_clean
from attack.Random_attack import prepare_data_with_random_attack
from attack.Average_attack import prepare_data_with_average_attack
from attack.Sybil_Attack import prepare_data_with_sybil_attack
import torch
import random
import numpy as np
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(42)
def split_data(data, test_size=0.3, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state)

def attack_data_deal(normal_file,attack_file1,attack_file2,attack_file3,proportion):
    user_item_pairs = np.loadtxt(normal_file, dtype=int, delimiter=',')
    num_users = np.max(user_item_pairs[:, 0]) + 1
    num_items = 3952
    rating_matrix = np.zeros((num_users, num_items))
    for uid, sid, rathing in user_item_pairs:
        rating_matrix[uid, sid] = rathing
    ones_column = np.zeros((rating_matrix.shape[0], 1))
    rating_matrix = np.hstack([rating_matrix, ones_column])
    malicious_user_percentage = 0.2
    filler_percentage = 0.0418
    data_with_attack1 = prepare_data_with_random_attack(rating_matrix, malicious_user_percentage, filler_percentage)
    data_with_attack2 = prepare_data_with_average_attack(rating_matrix, malicious_user_percentage, filler_percentage)
    data_with_attack3 = prepare_data_with_sybil_attack(rating_matrix, malicious_user_percentage, filler_percentage)
    data_with_attack4 = dataload(attack_file1)
    data_with_attack5 = dataload(attack_file2)
    data_with_attack6 = posion_data_clean(attack_file3,num_items)
    rating_matrix[rating_matrix > 0] = 1
    train_data, test_data = split_data(rating_matrix)
    num_rows_rating = test_data.shape[0]

    train_data_with_attack1, test_data_with_attack1 = split_data(data_with_attack1)
    num_rows_attack1 = test_data_with_attack1.shape[0]
    num_selected_rows1 = min(int(num_rows_rating * proportion), num_rows_attack1)
    selected_data1 = test_data_with_attack1[np.random.choice(num_rows_attack1, num_selected_rows1, replace=False), :]

    train_data_with_attack2, test_data_with_attack2 = split_data(data_with_attack2)
    num_rows_attack2 = test_data_with_attack2.shape[0]
    num_selected_rows2 = min(int(num_rows_rating * proportion), num_rows_attack2)
    selected_data2 = test_data_with_attack2[np.random.choice(num_rows_attack2, num_selected_rows2, replace=False), :]

    train_data_with_attack3, test_data_with_attack3 = split_data(data_with_attack3)
    num_rows_attack3 = test_data_with_attack3.shape[0]
    num_selected_rows3 = min(int(num_rows_rating * proportion), num_rows_attack3)
    selected_data3 = test_data_with_attack3[np.random.choice(num_rows_attack3, num_selected_rows3, replace=False), :]

    train_data_with_attack4, test_data_with_attack4 = split_data(data_with_attack4)
    train_data_with_attack5, test_data_with_attack5 = split_data(data_with_attack5)
    num_rows_attack4 = test_data_with_attack4.shape[0]
    num_rows_attack5 = test_data_with_attack5.shape[0]
    num_selected_rows4 = min(int(num_rows_rating * proportion), num_rows_attack4)
    num_selected_rows5 = min(int(num_rows_rating * proportion), num_rows_attack5)
    selected_data4 = test_data_with_attack4[np.random.choice(num_rows_attack4, num_selected_rows4, replace=False), :]
    selected_data5 = test_data_with_attack5[np.random.choice(num_rows_attack5, num_selected_rows5, replace=False), :]

    train_data_with_attack6, test_data_with_attack6 = split_data(data_with_attack6)
    num_rows_attack6 = test_data_with_attack6.shape[0]
    num_selected_rows6 = min(int(num_rows_rating * proportion), num_rows_attack6)
    selected_data6 = test_data_with_attack6[np.random.choice(num_rows_attack6, num_selected_rows6, replace=False), :]
    all_train_data_combined = np.concatenate(
        [
            train_data,
            train_data_with_attack1,
            train_data_with_attack2,
            train_data_with_attack3,
            train_data_with_attack4,
            train_data_with_attack5,
            train_data_with_attack6
        ], axis=0
    )
    all_test_data_combined1 = np.concatenate(
        [
            test_data,
            selected_data1,
        ], axis=0
    )
    all_test_data_combined2 = np.concatenate(
        [
            test_data,
            selected_data2,
        ], axis=0
    )
    all_test_data_combined3 = np.concatenate(
        [
            test_data,
            selected_data3,
        ], axis=0
    )
    all_test_data_combined4 = np.concatenate(
        [
            test_data,
            selected_data4,
        ], axis=0
    )
    all_test_data_combined5 = np.concatenate(
        [
            test_data,
            selected_data5,
        ], axis=0
    )
    all_test_data_combined6 = np.concatenate(
        [
            test_data,
            selected_data6,
        ], axis=0
    )
    np.random.shuffle(all_train_data_combined)
    np.random.shuffle(all_test_data_combined1)
    np.random.shuffle(all_test_data_combined2)
    np.random.shuffle(all_test_data_combined3)
    np.random.shuffle(all_test_data_combined4)
    np.random.shuffle(all_test_data_combined5)
    np.random.shuffle(all_test_data_combined6)
    return all_train_data_combined,all_test_data_combined1,all_test_data_combined2,all_test_data_combined3,all_test_data_combined4,all_test_data_combined5,all_test_data_combined6

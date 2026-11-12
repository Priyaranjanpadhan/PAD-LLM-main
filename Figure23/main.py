import os
import torch
from cnn_model import CNN3DModel
from data_clean.amazon_attack_data_clean import attack_data_deal
from util1 import shap_tag_heatmap_20users_from_cnn
from utils import  amazon_item
def load_model(model, filename):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()
        print(f"Model loaded from {filename}.")
    else:
        print(f"Model file {filename} not found. Initializing training from scratch.")

def main():
    normal_file = '../data/amazon/user_item.dat'
    features_file = '../data/amazon/item_category.dat'
    features_file2 = '../data/amazon/item_brand.dat'
    model_filename = "../model/amazon/3D_CNN.pth"
    attack_file1='../attack_data/amazon/Sur-ItemAE_fake_data_best.npz'
    attack_file2 ='../attack_data/amazon/Sur-WeightedMF-sgd_fake_data_best.npz'
    attack_file3 = '../attack_data/amazon/posion_data.txt'
    item_number = 2753
    item_label_number = 16
    performer_heads = 4
    proportion = 0.01

    item_labels = amazon_item(features_file,features_file2, item_number)
    all_train_data_combined,all_test_data_combined1,all_test_data_combined2,all_test_data_combined3,all_test_data_combined4,all_test_data_combined5,all_test_data_combined6 = attack_data_deal(normal_file,attack_file1,attack_file2,attack_file3,proportion)
    item_labels_tensor = torch.tensor(item_labels, dtype=torch.float32)
    rating_test1 = all_test_data_combined1[:, :-1]
    y_test1 = all_test_data_combined1[:, -1]
    rating_test_tensor1 = torch.tensor(rating_test1, dtype=torch.float32)
    y_test_tensor1 = torch.tensor(y_test1, dtype=torch.float32).view(-1, 1)
    device = torch.device("cuda")
    cnn_model = CNN3DModel(device, item_number, item_label_number, performer_heads)
    load_model(cnn_model, model_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device)

    shap_tag_heatmap_20users_from_cnn(
        cnn_model=cnn_model,
        rating_tensor=rating_test_tensor1,
        item_labels_tensor=item_labels_tensor,
        y_tensor=y_test_tensor1,
        bg_size=64,
        pick_mode="typical_by_logit",
        n_total=20,
        tag="Figure23",
        aggregate_only_rated=True,
        normalize_per_user=False
    )



if __name__ == "__main__":
    main()

import os
import torch
from cnn_model import train_cnn_3d, CNN3DModel
from data_clean.amazon_attack_data_clean import attack_data_deal
from detection_model import DetectionModel, train_detection_model, test_detection_model
from utils import extract_features, amazon_item
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
# 加载模型
def load_model(model, filename):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()
        print(f"Model loaded from {filename}.")
    else:
        print(f"Model file {filename} not found. Initializing training from scratch.")

def main():
    normal_file = 'data/amazon/user_item.dat'
    features_file = 'data/amazon/item_category.dat'
    features_file2 = 'data/amazon/item_brand.dat'
    model_filename = "model/amazon/3D_CNN.pth"
    attack_file1='attack_data/amazon/Sur-ItemAE_fake_data_best.npz'
    attack_file2 ='attack_data/amazon/Sur-WeightedMF-sgd_fake_data_best.npz'
    attack_file3 = 'attack_data/amazon/posion_data.txt'
    threshold = 0.9972
    item_number = 2753
    item_label_number = 16
    performer_heads = 4
    proportion = 0.1
    learning_rate = 0.0001

    item_labels = amazon_item(features_file,features_file2, item_number)
    all_train_data_combined,all_test_data_combined1,all_test_data_combined2,all_test_data_combined3,all_test_data_combined4,all_test_data_combined5,all_test_data_combined6 = attack_data_deal(normal_file,attack_file1,attack_file2,attack_file3,proportion)
    rating_train = all_train_data_combined[:, :-1]
    y_train = all_train_data_combined[:, -1]
    rating_train_tensor = torch.tensor(rating_train, dtype=torch.float32)
    item_labels_tensor = torch.tensor(item_labels, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    rating_test1 = all_test_data_combined1[:, :-1]
    y_test1 = all_test_data_combined1[:, -1]
    rating_test2 = all_test_data_combined2[:, :-1]
    y_test2 = all_test_data_combined2[:, -1]
    rating_test3 = all_test_data_combined3[:, :-1]
    y_test3 = all_test_data_combined3[:, -1]
    rating_test4 = all_test_data_combined4[:, :-1]
    y_test4 = all_test_data_combined4[:, -1]
    rating_test5 = all_test_data_combined5[:, :-1]
    y_test5 = all_test_data_combined5[:, -1]
    rating_test6 = all_test_data_combined6[:, :-1]
    y_test6 = all_test_data_combined6[:, -1]
    rating_test_tensor1 = torch.tensor(rating_test1, dtype=torch.float32)
    y_test_tensor1 = torch.tensor(y_test1, dtype=torch.float32).view(-1, 1)
    rating_test_tensor2 = torch.tensor(rating_test2, dtype=torch.float32)
    y_test_tensor2 = torch.tensor(y_test2, dtype=torch.float32).view(-1, 1)
    rating_test_tensor3 = torch.tensor(rating_test3, dtype=torch.float32)
    y_test_tensor3 = torch.tensor(y_test3, dtype=torch.float32).view(-1, 1)
    rating_test_tensor4 = torch.tensor(rating_test4, dtype=torch.float32)
    y_test_tensor4 = torch.tensor(y_test4, dtype=torch.float32).view(-1, 1)
    rating_test_tensor5 = torch.tensor(rating_test5, dtype=torch.float32)
    y_test_tensor5 = torch.tensor(y_test5, dtype=torch.float32).view(-1, 1)
    rating_test_tensor6 = torch.tensor(rating_test6, dtype=torch.float32)
    y_test_tensor6 = torch.tensor(y_test6, dtype=torch.float32).view(-1, 1)
    device = torch.device("cuda")
    cnn_model = CNN3DModel(device, item_number, item_label_number, performer_heads)
    load_model(cnn_model, model_filename)
    if not os.path.exists(model_filename):
        print("Training the ConvNeXt + Performer + Gate model...")
        cnn_model = train_cnn_3d(
            cnn_model,
            rating_train_tensor,
            item_labels_tensor,
            y_train_tensor,
            learning_rate,
            epochs=50
        )
        save_model(cnn_model, model_filename)
    else:
        print("Using the pre-loaded ConvNeXt + Performer + Gate model.")
    train_features = extract_features(cnn_model, rating_train_tensor, item_labels_tensor, device,)
    random_attack_test_features = extract_features(cnn_model, rating_test_tensor1, item_labels_tensor, device,)
    average_attack_features = extract_features(cnn_model, rating_test_tensor2, item_labels_tensor, device, )
    sybil_attack_features = extract_features(cnn_model, rating_test_tensor3, item_labels_tensor, device, )
    adversarial_attack_ItemAE_features = extract_features(cnn_model, rating_test_tensor4, item_labels_tensor, device, )
    adversarial_attack_WeightedMF_sgd_features = extract_features(cnn_model, rating_test_tensor5, item_labels_tensor, device, )
    posion_attack_features = extract_features(cnn_model, rating_test_tensor6, item_labels_tensor, device, )
    torch.cuda.empty_cache()
    detection_model = DetectionModel().to("cuda")
    train_detection_model(detection_model, train_features, y_train_tensor)
    print("Random Attack Results")
    test_detection_model(detection_model, random_attack_test_features , y_test_tensor1, threshold)
    print("Average Attack Results")
    test_detection_model(detection_model, average_attack_features, y_test_tensor2, threshold)
    print("Bandwagon Attack Results")
    test_detection_model(detection_model, sybil_attack_features, y_test_tensor3, threshold)
    print("ItemAE Attack Results")
    test_detection_model(detection_model, adversarial_attack_ItemAE_features, y_test_tensor4, threshold)
    print("WRMF+SGD Attack Results")
    test_detection_model(detection_model, adversarial_attack_WeightedMF_sgd_features, y_test_tensor5, threshold)
    print("GOAT Attack Results")
    test_detection_model(detection_model, posion_attack_features, y_test_tensor6, threshold)

if __name__ == "__main__":
    main()

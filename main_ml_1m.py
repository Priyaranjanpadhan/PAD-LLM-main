import os
import torch
from cnn_model import train_cnn_3d, CNN3DModel
from data_clean.ml_1m_attack_data_clean import attack_data_deal
from detection_model import DetectionModel, train_detection_model, test_detection_model
from utils import extract_features, load_item_label_data

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
def load_model(model, filename, device):
    if os.path.exists(filename):
        model.load_state_dict(
            torch.load(filename, map_location=device, weights_only=True)
        )
        model.eval()
        print(f"Model loaded from {filename}.")
    else:
        print(f"Model file {filename} not found. Initializing training from scratch.")

def main():
    normal_file = 'data/ml-1m/user_item_rating.txt'
    features_file = 'data/ml-1m/movies.dat'
    # I change the file path so that in line about whether to choose this pretrained or we will train section, in load_mode(cnn_model, mode_filename) if not os.path.exists(model_filename:).. in line about 71 72 73 around that somewhere.
    # so that everytime it will use the attention pooling, that I fixed and not the baseline model.
    model_filename = "model/ml_1m/3D_CNN_pase1_pooled.pth"
    attack_file1='attack_data/ml-1m/Sur-ItemAE_fake_data_best.npz'
    attack_file2 ='attack_data/ml-1m/Sur-WeightedMF-sgd_fake_data_best.npz'
    attack_file3 = 'attack_data/ml-1m/posion_data.txt'

    threshold = 0.95
    item_number = 3952
    item_label_number = 16
    performer_heads = 4
    proportion = 0.01
    learning_rate = 0.0001

    print("\n[MAIN] Step 1: Preparing attack datasets...", flush = True)
    all_train_data_combined,all_test_data_combined1,all_test_data_combined2,all_test_data_combined3,all_test_data_combined4,all_test_data_combined5,all_test_data_combined6 = attack_data_deal(normal_file,attack_file1,attack_file2,attack_file3,proportion)
    print("[MAIN] Step 1 completed.", flush=True)

    print("[MAIN] Step 2: Loading T5 item embeddings...", flush=True)
    item_labels = load_item_label_data(features_file,item_number)
    print(f"[MAIN] Step 2 completed. Shape: {item_labels.shape}", flush=True)

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

    # mps for mac graphics card cuda for nvidia graphics card
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[MAIN] Running model on: {device}", flush=True)

    print("[MAIN] Step 3: Creating CNN model on", device, flush=True)
    cnn_model = CNN3DModel(device, item_number, item_label_number, performer_heads)
    print("[MAIN] CNN model created.", flush=True)

    print("[MAIN] Checking checkpoint...", flush=True)
    load_model(cnn_model, model_filename, device)
    if not os.path.exists(model_filename):
        print("[MAIN] Step 4: Starting CNN training...", flush=True)
        print("Training the ConvNeXt + Performer + Gate model...")
        cnn_model = train_cnn_3d(cnn_model, rating_train_tensor, item_labels_tensor, y_train_tensor, learning_rate, epochs=30 )
        print("[MAIN] CNN training completed.", flush=True)
        save_model(cnn_model, model_filename)
    else:
        print("[MAIN] Using the pre-loaded ConvNeXt + Performer + Gate model.", flush=True)

    print("[MAIN] Step 5: Extracting training features...", flush=True)
    train_features = extract_features(cnn_model, rating_train_tensor, item_labels_tensor, device,)
    print("[MAIN] Training features extracted.", flush=True)

    print("[MAIN] Step 6: Extracting random attack test features...", flush=True)
    random_attack_test_features = extract_features(cnn_model, rating_test_tensor1, item_labels_tensor, device,)
    print("[MAIN] Random attack features extracted.", flush=True)

    print("[MAIN] Step 7: Extracting avearge attack features...", flush=True)
    average_attack_features = extract_features(cnn_model, rating_test_tensor2, item_labels_tensor, device, )
    print("[Main] Avearage attack features extracted.", flush=True)

    print("[MAIN] Step 8: Extracting sybil attack features...", flush=True)
    sybil_attack_features = extract_features(cnn_model, rating_test_tensor3, item_labels_tensor, device, )
    print("[MAIN] Sybil attack features extracted.", flush=True)

    print("[MAIN] Step 9: Extracting adversial attack ItemAE features...", flush=True)
    adversarial_attack_ItemAE_features = extract_features(cnn_model, rating_test_tensor4, item_labels_tensor, device, )
    print("[MAIN] Adversial attack ItemAE features extracted.", flush=True)

    print("[MAIN] Step 10: Extracting adversial attack WeightedMF_sgd features...", flush=True)
    adversarial_attack_WeightedMF_sgd_features = extract_features(cnn_model, rating_test_tensor5, item_labels_tensor, device, )
    print("[MAIN] Adversial attack WeightedMF_sgd features extracted.", flush=True)

    print("[MAIN] Step 10: Extracting poisoning attack features...", flush=True)
    posion_attack_features = extract_features(cnn_model, rating_test_tensor6, item_labels_tensor, device, )
    print("[MAIN] Poisoning attack features extracted.", flush=True)

    torch.mps.empty_cache()

    detection_model = DetectionModel().to(device)
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
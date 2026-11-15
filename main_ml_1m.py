import os
import argparse
import numpy as np
import torch
from cnn_model import train_cnn_3d, CNN3DModel
from data_clean.ml_1m_attack_data_clean import (
    attack_data_deal,
    create_loao_split,
)
from detection_model import DetectionModel, train_detection_model, test_detection_model
from utils import extract_features, load_item_label_data

LOAO_ATTACKS = (
    "Random",
    "Average",
    "Bandwagon",
    "ItemAE",
    "WRMF_SGD",
    "GOAT",
)

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


def _to_profile_tensors(data):
    """Separate a profile matrix into floating-point ratings and labels."""
    ratings = torch.tensor(data[:, :-1], dtype=torch.float32)
    labels = torch.tensor(data[:, -1], dtype=torch.float32).view(-1, 1)
    return ratings, labels


def save_detection_model(model, filename):
    """Save detector weights after its LazyLinear layer has been initialized."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"[LOAO] Detection model saved to {filename}", flush=True)


def load_detection_model(model, filename, input_features, device):
    """Initialize and restore a detector checkpoint for a completed LOAO fold."""
    if not os.path.exists(filename):
        return False
    # LazyLinear creates its first layer only after seeing one input batch.
    model(torch.zeros((1, input_features), device=device))
    model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))
    model.eval()
    print(f"[LOAO] Detection model loaded from {filename}", flush=True)
    return True


def run_loao_experiments():
    """Run all six Leave-One-Attack-Out zero-day experiments.

    Each fold creates new training data and a new pair of models. The attack
    family held out in a fold is never used for training, but it is evaluated
    with the other five families after training finishes.
    """
    normal_file = "data/ml-1m/user_item_rating.txt"
    features_file = "data/ml-1m/movies.dat"
    attack_file1 = "attack_data/ml-1m/Sur-ItemAE_fake_data_best.npz"
    attack_file2 = "attack_data/ml-1m/Sur-WeightedMF-sgd_fake_data_best.npz"
    attack_file3 = "attack_data/ml-1m/posion_data.txt"
    loao_root = "/content/drive/MyDrive/PAD-LLM_main/LOAO"
    item_number = 3952
    item_label_number = 16
    performer_heads = 4
    proportion = 0.01
    learning_rate = 0.0001
    threshold = 0.95

    # Item metadata is independent of attack type, so the cached T5 embeddings
    # can be computed once and reused by all six fresh CNN models.
    item_labels = load_item_label_data(features_file, item_number)
    item_labels_tensor = torch.tensor(item_labels, dtype=torch.float32)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[LOAO] Running models on: {device}", flush=True)

    for held_out_attack in LOAO_ATTACKS:
        print(
            f"\n[LOAO] Starting fold: held out {held_out_attack}",
            flush=True,
        )
        split = create_loao_split(
            normal_file,
            attack_file1,
            attack_file2,
            attack_file3,
            held_out_attack=held_out_attack,
            proportion=proportion,
        )
        train_ratings, train_labels = _to_profile_tensors(split["train"])
        fold_dir = os.path.join(loao_root, held_out_attack)
        cnn_filename = os.path.join(fold_dir, "cnn_final.pth")
        cnn_checkpoint = os.path.join(fold_dir, "cnn_checkpoint.pth")
        detector_filename = os.path.join(fold_dir, "detector_final.pth")

        # A new CNN is required for every fold; loading the closed-set model
        # would expose the model to attack families that should be unseen.
        cnn_model = CNN3DModel(
            device, item_number, item_label_number, performer_heads
        )
        if os.path.exists(cnn_filename):
            load_model(cnn_model, cnn_filename, device)
        else:
            cnn_model = train_cnn_3d(
                cnn_model,
                train_ratings,
                item_labels_tensor,
                train_labels,
                learning_rate,
                epochs=30,
                checkpoint_path=cnn_checkpoint,
            )
            save_model(cnn_model, cnn_filename)

        # Features are extracted from the newly trained fold-specific CNN.
        train_features = extract_features(
            cnn_model, train_ratings, item_labels_tensor, device
        )
        detector = DetectionModel().to(device)
        detector_loaded = load_detection_model(
            detector, detector_filename, train_features.shape[1], device
        )
        if not detector_loaded:
            train_detection_model(
                detector,
                train_features,
                train_labels,
                device=device,
            )
            save_detection_model(detector, detector_filename)

        print(
            f"[LOAO] Results for fold holding out {held_out_attack}:",
            flush=True,
        )
        for attack_name in LOAO_ATTACKS:
            test_ratings, test_labels = _to_profile_tensors(
                split["test_by_attack"][attack_name]
            )
            test_features = extract_features(
                cnn_model, test_ratings, item_labels_tensor, device
            )
            metrics = test_detection_model(
                detector,
                test_features,
                test_labels,
                threshold,
                device=device,
            )
            print(
                f"[LOAO] {attack_name}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}",
                flush=True,
            )

def main():
    normal_file = 'data/ml-1m/user_item_rating.txt'
    features_file = 'data/ml-1m/movies.dat'
    # I change the file path so that in line about whether to choose this pretrained or we will train section, in load_mode(cnn_model, mode_filename) if not os.path.exists(model_filename:).. in line about 71 72 73 around that somewhere.
    # so that everytime it will use the attention pooling, that I fixed and not the baseline model.
    # This model_filename is a mere placeholder for the leraning file and right now it will live on the drive so it is looking like that. and the model folder in the PAD-LLM-main is not of use now, because the TPU will directly take the model
    # from the drive itself. 
    model_filename = "/content/drive/MyDrive/PAD-LLM_main/3D_CNN_attention_pool.pth"
    # The final model is used for inference; this separate file preserves training
    # progress and lets cnn_model.py resume after a Colab runtime interruption.
    checkpoint_filename = (
        "/content/drive/MyDrive/PAD-LLM_main/3D_CNN_attention_pool.checkpoint.pth"
    )
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

    # The following detailed audit is kept commented out because the current
    # investigation only needs the training-label distribution and class weights.
    # Uncomment it when you need to inspect rating ranges and values for every split.
    # def inspect_dataset(name, data):
    #     labels = data[:, -1]
    #     ratings = data[:, :-1]

    #     print(f"\n========== {name} ==========")
    #     print("Total users:", len(data))
    #     print("Genuine users:", np.sum(labels == 0))
    #     print("Malicious users:", np.sum(labels == 1))

    #     for label, user_type in [(0, "Genuine"), (1, "Malicious")]:
    #         selected = ratings[labels == label]

    #         if len(selected) > 0:
    #             print(f"{user_type} rating range:",
    #                   selected.min(), "to", selected.max())

    #             print(f"{user_type} unique values:",
    #                   np.unique(selected)[:15])

    # inspect_dataset("TRAINING DATA", all_train_data_combined)
    # inspect_dataset("RANDOM TEST", all_test_data_combined1)
    # inspect_dataset("AVERAGE TEST", all_test_data_combined2)
    # inspect_dataset("BANDWAGON TEST", all_test_data_combined3)
    # inspect_dataset("ITEMAE TEST", all_test_data_combined4)
    # inspect_dataset("WRMF TEST", all_test_data_combined5)
    # inspect_dataset("GOAT TEST", all_test_data_combined6)
    # print("Dataset inspection complete.")
    # return

    # ================= DATA DISTRIBUTION AUDIT =================
    # This audit checks how many genuine and malicious profiles are in the
    # training set, then compares the repository's positive-class weight with
    # the conventional inverse-frequency weight used for imbalanced classes.

    # =======================================================================

    # train_labels = all_train_data_combined[:, -1]

    # total_users = len(train_labels)
    # genuine_users = int((train_labels == 0).sum())
    # malicious_users = int((train_labels == 1).sum())

    # current_pos_weight = total_users / (2 * malicious_users)
    # standard_pos_weight = genuine_users / malicious_users

    # print("\n========== TRAINING DISTRIBUTION AUDIT ==========")
    # print("Total profiles:", total_users)
    # print("Genuine profiles:", genuine_users)
    # print("Malicious profiles:", malicious_users)

    # print(
    #     "Malicious percentage:",
    #     round(malicious_users / total_users * 100, 2), "%"
    # )

    # print("\n========== CLASS WEIGHT AUDIT ==========")
    # print("Current repository pos_weight:", current_pos_weight)
    # print("Conventional inverse-frequency pos_weight:", standard_pos_weight)

    # raise SystemExit("Dataset audit completed. Stopping before CNN training.")

    # # ========================================================================

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
        cnn_model = train_cnn_3d(
            cnn_model,
            rating_train_tensor,
            item_labels_tensor,
            y_train_tensor,
            learning_rate,
            epochs=30,
            checkpoint_path=checkpoint_filename,
        )
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

    # as the structure shifted from mps to tpu, we need to check where t5 embeddings will live. 
    # previously it was only torch.mps.empty_cache() but cuda is also possiblity.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
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
    # Keep the original closed-set experiment as the default. Use --loao only
    # when deliberately starting the six-fold zero-day evaluation.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loao",
        action="store_true",
        help="run the six Leave-One-Attack-Out experiments",
    )
    args = parser.parse_args()
    if args.loao:
        run_loao_experiments()
    else:
        main()
"""Data preparation utilities for MovieLens-1M poisoning-attack experiments.

This file contains two separate data-preparation paths:

1. attack_data_deal(...)
   The original closed-set experiment: train on all six attack families and
   create one test set per attack family.

2. create_loao_split(...)
   The Phase 2 Leave-One-Attack-Out (LOAO) experiment: exclude one complete
   attack family from training and retain it only for testing.

The original attack_data_deal() workflow is intentionally preserved so that
existing closed-set experiments continue to work.
"""

import os
import sys

# When this file is executed directly, Python starts its import search inside
# data_clean/. Add the repository root so sibling packages such as attack/
# remain importable; normal package imports are unaffected.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from attack.Adversarial_attack_data_clean import dataload
from attack.Posion_data_clean import posion_data_clean
from attack.Random_attack import prepare_data_with_random_attack
from attack.Average_attack import prepare_data_with_average_attack
from attack.Sybil_Attack import prepare_data_with_sybil_attack

import torch
import random
import numpy as np


# ---------------------------------------------------------------------------
# 1. Reproducibility utilities
# ---------------------------------------------------------------------------

def set_random_seed(seed=42):
    """Set Python, NumPy, and PyTorch random seeds.

    This helps make attack generation, sampling, and model-side randomness
    reproducible. CUDA-specific settings are applied only when CUDA is present.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Preserve the original module-level seed initialization.
# The LOAO function also resets the seed when it is called, so that each fold
# with the same seed uses the same data split and attack sampling.
set_random_seed(42)


def split_data(data, test_size=0.3, random_state=42):
    """Split a profile matrix into training and testing portions."""
    return train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# 2. Original closed-set data preparation
# ---------------------------------------------------------------------------

def attack_data_deal(
    normal_file,
    attack_file1,
    attack_file2,
    attack_file3,
    proportion,
):
    """Prepare the original closed-set experiment data.

    All six attack families are included in the training data. For testing,
    this function creates six separate test sets. Each test set contains the
    same genuine test users plus a sample from one attack family.

    This function is retained for Phase 1 / closed-set comparison. Do not use
    it to run a Leave-One-Attack-Out experiment.
    """

    # Load genuine user-item-rating triplets: user_id, item_id, rating.
    user_item_pairs = np.loadtxt(
        normal_file,
        dtype=int,
        delimiter=",",
    )

    num_users = np.max(user_item_pairs[:, 0]) + 1
    num_items = 3952

    # Reconstruct the genuine user-item rating matrix.
    rating_matrix = np.zeros((num_users, num_items))
    for uid, sid, rating in user_item_pairs:
        rating_matrix[uid, sid] = rating

    # The final column stores the class label:
    # 0 = genuine user, 1 = malicious user.
    ones_column = np.zeros((rating_matrix.shape[0], 1))
    rating_matrix = np.hstack([rating_matrix, ones_column])

    # Generate the three heuristic attack families.
    malicious_user_percentage = 0.2
    filler_percentage = 0.0418

    data_with_attack1 = prepare_data_with_random_attack(
        rating_matrix,
        malicious_user_percentage,
        filler_percentage,
    )
    data_with_attack2 = prepare_data_with_average_attack(
        rating_matrix,
        malicious_user_percentage,
        filler_percentage,
    )
    data_with_attack3 = prepare_data_with_sybil_attack(
        rating_matrix,
        malicious_user_percentage,
        filler_percentage,
    )

    # Load the two adversarially learned attack families and GOAT profiles.
    data_with_attack4 = dataload(attack_file1)                  # ItemAE
    data_with_attack5 = dataload(attack_file2)                  # WRMF+SGD
    data_with_attack6 = posion_data_clean(attack_file3, num_items)  # GOAT

    # Convert genuine ratings into binary interaction indicators:
    # 0 = no interaction, 1 = interacted with the item.
    rating_matrix[rating_matrix > 0] = 1

    # Split the genuine profiles. These same genuine test profiles are used
    # in all six attack-specific test sets.
    train_data, test_data = split_data(rating_matrix)
    num_rows_rating = test_data.shape[0]

    # ----------------------- Random attack -----------------------
    train_data_with_attack1, test_data_with_attack1 = split_data(
        data_with_attack1
    )
    num_rows_attack1 = test_data_with_attack1.shape[0]
    num_selected_rows1 = min(
        int(num_rows_rating * proportion),
        num_rows_attack1,
    )
    selected_data1 = test_data_with_attack1[
        np.random.choice(num_rows_attack1, num_selected_rows1, replace=False),
        :,
    ]

    # ----------------------- Average attack ----------------------
    train_data_with_attack2, test_data_with_attack2 = split_data(
        data_with_attack2
    )
    num_rows_attack2 = test_data_with_attack2.shape[0]
    num_selected_rows2 = min(
        int(num_rows_rating * proportion),
        num_rows_attack2,
    )
    selected_data2 = test_data_with_attack2[
        np.random.choice(num_rows_attack2, num_selected_rows2, replace=False),
        :,
    ]

    # ---------------------- Bandwagon attack ---------------------
    train_data_with_attack3, test_data_with_attack3 = split_data(
        data_with_attack3
    )
    num_rows_attack3 = test_data_with_attack3.shape[0]
    num_selected_rows3 = min(
        int(num_rows_rating * proportion),
        num_rows_attack3,
    )
    selected_data3 = test_data_with_attack3[
        np.random.choice(num_rows_attack3, num_selected_rows3, replace=False),
        :,
    ]

    # ------------------------ ItemAE attack -----------------------
    train_data_with_attack4, test_data_with_attack4 = split_data(
        data_with_attack4
    )
    num_rows_attack4 = test_data_with_attack4.shape[0]
    num_selected_rows4 = min(
        int(num_rows_rating * proportion),
        num_rows_attack4,
    )
    selected_data4 = test_data_with_attack4[
        np.random.choice(num_rows_attack4, num_selected_rows4, replace=False),
        :,
    ]

    # ----------------------- WRMF+SGD attack ---------------------
    train_data_with_attack5, test_data_with_attack5 = split_data(
        data_with_attack5
    )
    num_rows_attack5 = test_data_with_attack5.shape[0]
    num_selected_rows5 = min(
        int(num_rows_rating * proportion),
        num_rows_attack5,
    )
    selected_data5 = test_data_with_attack5[
        np.random.choice(num_rows_attack5, num_selected_rows5, replace=False),
        :,
    ]

    # -------------------------- GOAT attack -----------------------
    train_data_with_attack6, test_data_with_attack6 = split_data(
        data_with_attack6
    )
    num_rows_attack6 = test_data_with_attack6.shape[0]
    num_selected_rows6 = min(
        int(num_rows_rating * proportion),
        num_rows_attack6,
    )
    selected_data6 = test_data_with_attack6[
        np.random.choice(num_rows_attack6, num_selected_rows6, replace=False),
        :,
    ]

    # Closed-set training includes genuine profiles and ALL six attack types.
    all_train_data_combined = np.concatenate(
        [
            train_data,
            train_data_with_attack1,
            train_data_with_attack2,
            train_data_with_attack3,
            train_data_with_attack4,
            train_data_with_attack5,
            train_data_with_attack6,
        ],
        axis=0,
    )

    # Each test set contains genuine users plus one attack family.
    all_test_data_combined1 = np.concatenate(
        [test_data, selected_data1], axis=0
    )
    all_test_data_combined2 = np.concatenate(
        [test_data, selected_data2], axis=0
    )
    all_test_data_combined3 = np.concatenate(
        [test_data, selected_data3], axis=0
    )
    all_test_data_combined4 = np.concatenate(
        [test_data, selected_data4], axis=0
    )
    all_test_data_combined5 = np.concatenate(
        [test_data, selected_data5], axis=0
    )
    all_test_data_combined6 = np.concatenate(
        [test_data, selected_data6], axis=0
    )

    # Shuffle rows so that genuine and malicious profiles are mixed.
    np.random.shuffle(all_train_data_combined)
    np.random.shuffle(all_test_data_combined1)
    np.random.shuffle(all_test_data_combined2)
    np.random.shuffle(all_test_data_combined3)
    np.random.shuffle(all_test_data_combined4)
    np.random.shuffle(all_test_data_combined5)
    np.random.shuffle(all_test_data_combined6)

    return (
        all_train_data_combined,
        all_test_data_combined1,
        all_test_data_combined2,
        all_test_data_combined3,
        all_test_data_combined4,
        all_test_data_combined5,
        all_test_data_combined6,
    )


# ---------------------------------------------------------------------------
# 3. Phase 2: Leave-One-Attack-Out (LOAO) data preparation
# ---------------------------------------------------------------------------

def create_loao_split(
    normal_file,
    attack_file1,
    attack_file2,
    attack_file3,
    held_out_attack="GOAT",
    proportion=0.01,
    seed=42,
):
    """Create one Leave-One-Attack-Out training/testing split.

    Example: held_out_attack="GOAT"

    Training data:
        Genuine users + Random + Average + Bandwagon + ItemAE + WRMF+SGD
        GOAT training profiles are excluded completely.

    Testing data:
        Genuine test users + selected profiles from all six attack families.

    The returned test_by_attack dictionary contains six separate test sets.
    For example, test_by_attack["GOAT"] contains genuine test users plus
    GOAT test profiles. This allows the held-out attack's performance to be
    measured separately from the five known attacks.

    IMPORTANT:
        Call this function separately for each held-out attack family. Each
        fold must train a fresh CNN and a fresh detection classifier. Do not
        load the CNN checkpoint from the closed-set experiment.
    """

    # Reset the random seeds at the start of every fold.
    # With the same seed, every LOAO fold receives the same underlying data
    # splits and selected attack samples. The only intended difference is
    # which attack family is excluded from the training set.
    set_random_seed(seed)

    attack_names = (
        "Random",
        "Average",
        "Bandwagon",
        "ItemAE",
        "WRMF_SGD",
        "GOAT",
    )

    # Validate the requested held-out attack name before doing expensive work.
    if held_out_attack not in attack_names:
        raise ValueError(
            f"held_out_attack must be one of {attack_names}; "
            f"received {held_out_attack!r}."
        )

    # The requested attack ratio must be positive and smaller than or equal
    # to 1. For example, 0.01 means 1%, and 0.05 means 5%.
    if not (0 < proportion <= 1):
        raise ValueError(
            "proportion must be greater than 0 and less than or equal to 1."
        )

    # ------------------- Load genuine user profiles -------------------

    # Each input row contains: user_id, item_id, rating.
    # ndmin=2 ensures that the result is still a 2D array if the file ever
    # contains only one row.
    user_item_pairs = np.loadtxt(
        normal_file,
        dtype=int,
        delimiter=",",
        ndmin=2,
    )

    num_users = np.max(user_item_pairs[:, 0]) + 1
    num_items = 3952

    # Reconstruct the original rating matrix.
    rating_matrix = np.zeros((num_users, num_items))
    for uid, sid, rating in user_item_pairs:
        rating_matrix[uid, sid] = rating

    # Append the genuine-user label column. Genuine users have label 0.
    rating_matrix = np.hstack(
        [rating_matrix, np.zeros((rating_matrix.shape[0], 1))]
    )

    # ---------------------- Generate attack families ----------------------

    malicious_user_percentage = 0.2
    filler_percentage = 0.0418

    # Keep each attack family in a separate dictionary entry. This is
    # essential: if we concatenate attacks too early, we lose the information
    # needed to exclude one complete family from training.
    attack_pools = {
        "Random": prepare_data_with_random_attack(
            rating_matrix,
            malicious_user_percentage,
            filler_percentage,
        ),
        "Average": prepare_data_with_average_attack(
            rating_matrix,
            malicious_user_percentage,
            filler_percentage,
        ),
        "Bandwagon": prepare_data_with_sybil_attack(
            rating_matrix,
            malicious_user_percentage,
            filler_percentage,
        ),
        "ItemAE": dataload(attack_file1),
        "WRMF_SGD": dataload(attack_file2),
        "GOAT": posion_data_clean(attack_file3, num_items),
    }

    # ----------------------- Prepare genuine split -----------------------

    # Convert only the item-rating columns into binary interactions.
    # The final label column must remain 0 for genuine users.
    binary_ratings = rating_matrix.copy()
    binary_ratings[:, :-1] = (binary_ratings[:, :-1] > 0).astype(float)

    # Use the same random_state for every fold with this seed. Therefore,
    # the genuine training and test users remain identical across the six
    # held-out-attack experiments for a given seed.
    genuine_train, genuine_test = split_data(
        binary_ratings,
        random_state=seed,
    )

    # ----------------------- Split each attack family -----------------------

    attack_train = {}
    attack_test = {}
    selected_attack_test = {}

    for attack_name in attack_names:
        attack_data = attack_pools[attack_name]

        # Split this attack family's profiles into its own train/test pools.
        # The held-out family's training portion will be deliberately omitted
        # from the final LOAO training data below.
        attack_train[attack_name], attack_test[attack_name] = split_data(
            attack_data,
            random_state=seed,
        )

        test_pool = attack_test[attack_name]

        if len(test_pool) == 0:
            raise ValueError(
                f"The test pool for attack {attack_name} is empty. "
                "Cannot perform the requested LOAO evaluation."
            )

        # Select approximately 'proportion' malicious profiles relative to
        # the number of genuine test users. round() avoids systematically
        # rounding every fractional count down. At least one attack profile
        # is selected when the pool is non-empty.
        requested_count = max(
            1,
            round(len(genuine_test) * proportion),
        )
        selected_count = min(requested_count, len(test_pool))

        # Randomly select test profiles without replacement, so the same
        # profile cannot appear twice in this particular attack test sample.
        selected_indices = np.random.choice(
            len(test_pool),
            selected_count,
            replace=False,
        )
        selected_attack_test[attack_name] = test_pool[selected_indices]

    # ------------------------- Build training data -------------------------

    # The training set starts with genuine users.
    training_parts = [genuine_train]

    # Add the training profiles from the five known attack families.
    # The held-out attack is intentionally skipped here. Its profiles must
    # not be used to train either the CNN feature extractor or the final
    # detection classifier.
    train_attack_families = [
        name for name in attack_names
        if name != held_out_attack
    ]

    for attack_name in train_attack_families:
        training_parts.append(attack_train[attack_name])

    train_data = np.concatenate(training_parts, axis=0)

    # -------------------------- Build test data ---------------------------

    # Each per-attack test set contains:
    #   (1) the same genuine test users, and
    #   (2) selected malicious profiles from exactly one attack family.
    # This is the cleanest way to report metrics for each attack separately.
    test_by_attack = {
        attack_name: np.concatenate(
            [genuine_test, selected_attack_test[attack_name]],
            axis=0,
        )
        for attack_name in attack_names
    }

    # The combined test set contains genuine users once, plus selected test
    # profiles from all six attack families. This can be used for an overall
    # mixed-attack evaluation. Note that its total attack ratio is therefore
    # approximately six times 'proportion', not 'proportion'.
    test_data = np.concatenate(
        [genuine_test]
        + [selected_attack_test[name] for name in attack_names],
        axis=0,
    )

    # Shuffle rows after assembling the data. Labels remain in the last column,
    # so shuffling does not break the connection between a profile and label.
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    for family_test_data in test_by_attack.values():
        np.random.shuffle(family_test_data)

    # Only expose training arrays for the five families that are allowed in
    # this fold. The held-out family's training pool is not returned.
    train_by_attack = {
        name: attack_train[name]
        for name in train_attack_families
    }

    # Return both the arrays used by the model and metadata used for auditing.
    return {
        "held_out_attack": held_out_attack,
        "seed": seed,
        "proportion": proportion,
        "train": train_data,
        "test": test_data,
        "test_by_attack": test_by_attack,
        "train_by_attack": train_by_attack,
        "train_attack_families": train_attack_families,
        "genuine_train": genuine_train,
        "genuine_test": genuine_test,
        "selected_attack_test": selected_attack_test,
        # This is a count only. The actual held-out training array is not
        # included in the returned training data or train_by_attack mapping.
        "excluded_attack_train_rows": len(attack_train[held_out_attack]),
    }


# ---------------------------------------------------------------------------
# 4. Standalone GOAT-held-out data audit
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fill in these paths before running this file directly.
    # If you normally call the function from main_ml_1m.py or a notebook,
    # you can copy the audit block below into that environment instead.
    normal_file = "data/ml-1m/user_item_rating.txt"
    attack_file1 = "attack_data/ml-1m/Sur-ItemAE_fake_data_best.npz"
    attack_file2 = "attack_data/ml-1m/Sur-WeightedMF-sgd_fake_data_best.npz"
    attack_file3 = "attack_data/ml-1m/posion_data.txt"

    data = create_loao_split(
        normal_file=normal_file,
        attack_file1=attack_file1,
        attack_file2=attack_file2,
        attack_file3=attack_file3,
        held_out_attack="GOAT",
        proportion=0.01,
        seed=42,
    )

    print("HELD-OUT ATTACK:", data["held_out_attack"])
    print("RANDOM SEED:", data["seed"])
    print("PER-ATTACK TEST PROPORTION:", data["proportion"])

    print("\nATTACK FAMILIES ALLOWED IN TRAINING:")
    print(data["train_attack_families"])

    print("\nHELD-OUT ATTACK TRAINING ROWS EXCLUDED:")
    print(data["excluded_attack_train_rows"])

    print("\nTRAINING DATA")
    print("Shape:", data["train"].shape)
    print("Genuine:", int(np.sum(data["train"][:, -1] == 0)))
    print("Malicious:", int(np.sum(data["train"][:, -1] == 1)))

    print("\nCOMBINED TEST DATA")
    print("Shape:", data["test"].shape)
    print("Genuine:", int(np.sum(data["test"][:, -1] == 0)))
    print("Malicious:", int(np.sum(data["test"][:, -1] == 1)))

    print("\nINDIVIDUAL ATTACK TEST SETS")
    for attack_name, test_set in data["test_by_attack"].items():
        print(
            attack_name,
            "Shape:", test_set.shape,
            "Genuine:", int(np.sum(test_set[:, -1] == 0)),
            "Malicious:", int(np.sum(test_set[:, -1] == 1)),
        )

    # ----------------------------- Assertions -----------------------------
    # These assertions stop the run if the data structure does not match the
    # intended GOAT-held-out experimental design.

    expected_training_attacks = [
        "Random", "Average", "Bandwagon", "ItemAE", "WRMF_SGD"
    ]

    # Confirm that exactly the five non-GOAT attack families are available
    # for model training.
    assert data["train_attack_families"] == expected_training_attacks
    assert "GOAT" not in data["train_by_attack"]

    # Confirm that the combined training matrix contains genuine users plus
    # all profiles from the five permitted attack training pools.
    expected_train_rows = (
        len(data["genuine_train"])
        + sum(len(x) for x in data["train_by_attack"].values())
    )
    assert len(data["train"]) == expected_train_rows

    # Confirm that both the interaction features and class labels are binary.
    # (The final label column is excluded from the interaction-feature check.)
    assert np.all(np.isin(data["train"][:, :-1], [0, 1]))
    assert np.all(np.isin(data["train"][:, -1], [0, 1]))

    # Genuine users must have label 0; malicious training profiles label 1.
    assert np.all(data["genuine_train"][:, -1] == 0)
    for attack_name, attack_train_data in data["train_by_attack"].items():
        assert np.all(attack_train_data[:, -1] == 1), (
            f"Unexpected non-malicious label in {attack_name} training data."
        )

    # Confirm that the combined test data contains genuine users once and
    # the selected samples from all six attack families.
    expected_test_rows = (
        len(data["genuine_test"])
        + sum(len(x) for x in data["selected_attack_test"].values())
    )
    assert len(data["test"]) == expected_test_rows
    assert np.all(np.isin(data["test"][:, :-1], [0, 1]))
    assert np.all(np.isin(data["test"][:, -1], [0, 1]))

    # Every attack-specific test set must contain genuine profiles and only
    # the selected profiles from that one attack family.
    for attack_name, test_set in data["test_by_attack"].items():
        assert len(test_set) == (
            len(data["genuine_test"])
            + len(data["selected_attack_test"][attack_name])
        )
        assert np.sum(test_set[:, -1] == 0) == len(data["genuine_test"])
        assert np.sum(test_set[:, -1] == 1) == len(
            data["selected_attack_test"][attack_name]
        )

    # Confirm that the GOAT family is actually present in the held-out test
    # set even though it is excluded from training.
    assert len(data["selected_attack_test"]["GOAT"]) > 0
    assert np.sum(data["test_by_attack"]["GOAT"][:, -1] == 1) > 0

    print("\nGOAT LOAO DATA AUDIT PASSED")

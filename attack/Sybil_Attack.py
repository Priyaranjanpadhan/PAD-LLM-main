import numpy as np

def calculate_num_malicious_users(total_users, malicious_user_percentage):
    num_malicious_users = max(1, int(total_users * malicious_user_percentage))
    print(f"Number of malicious users: {num_malicious_users}")
    return num_malicious_users

def calculate_filler_items(total_items, filler_percentage):
    num_filler_items = max(1, int(total_items * filler_percentage))
    print(f"Number of filler items per malicious user: {num_filler_items}")
    return num_filler_items

def select_target_item_by_popularity(data):
    num_items = data.shape[1] - 1
    ratings = data[:, :num_items]
    popularity = np.sum(ratings > 0, axis=0)
    target_item = np.argmax(popularity)
    print(f"Selected target item: {target_item}, popularity: {popularity[target_item]}")
    return target_item

def select_popular_items(data, top_k=10, exclude_item=None):
    num_items = data.shape[1] - 1
    ratings = data[:, :num_items]
    popularity = np.sum(ratings > 0, axis=0)
    sorted_items = np.argsort(popularity)[::-1]
    if exclude_item is not None:
        sorted_items = sorted_items[sorted_items != exclude_item]
    popular_items = sorted_items[:top_k]
    print(f"Selected {len(popular_items)} popular items for Bandwagon attack.")
    return popular_items

def generate_bandwagon_attack_users_binary(
    data,
    num_malicious,
    num_filler_items,
    target_item,
    popular_items
):
    num_items = data.shape[1] - 1
    malicious_data = np.zeros((num_malicious, num_items + 1), dtype=np.float32)
    excluded_items = np.concatenate(([target_item], popular_items))
    candidate_filler_items = np.setdiff1d(np.arange(num_items), excluded_items)
    actual_filler_items = min(num_filler_items, len(candidate_filler_items))
    for i in range(num_malicious):
        malicious_data[i, target_item] = 5
        malicious_data[i, popular_items] = 5
        if actual_filler_items > 0:
            filler_items = np.random.choice(candidate_filler_items, size=actual_filler_items, replace=False)
            malicious_data[i, filler_items] = np.random.randint(1, 6, size=actual_filler_items)
        malicious_data[i, -1] = 1
    malicious_data[:, :-1] = np.where(malicious_data[:, :-1] > 0, 1, 0)
    print(f"Generated {num_malicious} Bandwagon attack users.")
    return malicious_data


def prepare_data_with_sybil_attack(
    data,
    malicious_user_percentage,
    filler_percentage,
    target_item=None,
    top_k_popular=10
):
    total_users = data.shape[0]
    total_items = data.shape[1] - 1
    num_malicious = calculate_num_malicious_users(total_users, malicious_user_percentage)
    num_filler_items = calculate_filler_items(total_items, filler_percentage)
    if target_item is None:
        target_item = select_target_item_by_popularity(data)
    popular_items = select_popular_items(data, top_k=top_k_popular, exclude_item=target_item)
    malicious_data = generate_bandwagon_attack_users_binary(
        data=data,
        num_malicious=num_malicious,
        num_filler_items=num_filler_items,
        target_item=target_item,
        popular_items=popular_items
    )
    return malicious_data
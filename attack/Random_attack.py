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
    item_popularity = np.sum(ratings > 0, axis=0)
    target_item = np.argmax(item_popularity)
    print(f"Selected target item: {target_item}, popularity: {item_popularity[target_item]}")
    return target_item

def generate_random_attack_users_binary(data, num_malicious, num_filler_items, target_item):
    num_items = data.shape[1] - 1
    malicious_data = np.zeros((num_malicious, num_items + 1), dtype=np.float32)
    candidate_items = np.delete(np.arange(num_items), target_item)
    actual_filler_items = min(num_filler_items, len(candidate_items))
    for i in range(num_malicious):
        filler_items = np.random.choice(candidate_items, size=actual_filler_items, replace=False)
        malicious_data[i, filler_items] = np.random.randint(1, 6, size=actual_filler_items)
        malicious_data[i, target_item] = 5
        malicious_data[i, -1] = 1
    malicious_data[:, :-1] = np.where(malicious_data[:, :-1] > 0, 1, 0)
    print(f"Generated {num_malicious} random attack users with {actual_filler_items} filler items each.")
    return malicious_data

def prepare_data_with_random_attack(data, malicious_user_percentage, filler_percentage):
    total_users = data.shape[0]
    total_items = data.shape[1] - 1
    num_malicious = calculate_num_malicious_users(total_users, malicious_user_percentage)
    num_filler_items = calculate_filler_items(total_items, filler_percentage)
    target_item = select_target_item_by_popularity(data)
    malicious_data = generate_random_attack_users_binary(
        data=data,
        num_malicious=num_malicious,
        num_filler_items=num_filler_items,
        target_item=target_item
    )
    return malicious_data
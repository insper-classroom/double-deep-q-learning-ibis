import pickle

def save_list_to_file(lst, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(lst, f)

def load_list_from_file(file_path):
    with open(file_path, 'rb') as f:
        lst = pickle.load(f)
    return lst

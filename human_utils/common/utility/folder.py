import os


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def change_tuple_element(t, dim, val):
    l = list(t)
    l[dim] = val
    return tuple(l)


def find_files_in_folder(folder_name, end_str):
    files = []
    for f in os.listdir(folder_name):
        if f.endswith(end_str):
            files.append(os.path.join(folder_name, f))
    files.sort()
    return files

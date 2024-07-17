from fnmatch import fnmatch
import os


def get_file_prefixes_from_path(data_path, criteria="*.bin"):
    prefixes = []

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name, criteria):
                prefixes.append(os.path.join(path, name)[:-(len(criteria)-1)])

    prefixes.sort()

    return prefixes

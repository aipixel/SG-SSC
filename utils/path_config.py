# Path configuration utility

import os
from utils import debug


def read_config(cfg_file="paths.conf"):
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), cfg_file)
    # cfg_path = os.path.abspath(cfg_file)
    cfg_file = os.path.basename(cfg_path)

    if not os.path.isfile(cfg_path):
        print('Path config file not found:', cfg_path)
        print('Please, edit provided example_paths.conf and save it as', cfg_file)
        raise Exception('Path config file not found.', cfg_path)

    path_dict = {}

    with(open(cfg_path, 'r')) as file:
        for line in file:
            ln = line.split()
            if (len(ln) == 0) or (ln[0][0:1] == '#'):
                continue
            if (len(ln) == 2) or (len(ln) > 2 and ln[0][0:1] == '#'):

                path_dict.update({ln[0]: ln[1]})

                debug.info('%-15s %s' % (ln[0]+':', ln[1]))
            else:
                raise Exception('Error in config %s:%s' % (cfg_path, line))

    return path_dict

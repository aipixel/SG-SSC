import os
import pandas as pd


def get_run():
    run_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "runcount.csv")
    if os.path.exists(run_file):
        rc = pd.read_csv(run_file)
    else:
        rc=pd.DataFrame(data={'run':['0']}, dtype='int')
    run = int(rc.run[0])
    run += 1
    rc.run = [run]
    rc.to_csv(run_file, index=False)
    return run

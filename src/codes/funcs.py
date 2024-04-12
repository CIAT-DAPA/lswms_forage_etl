import sys
import pandas as pd
from pandas.errors import EmptyDataError

def truncate_file(file_name):
    f = open(file_name, "w")
    f.truncate()
    f.close()
def exit_program():
    print("Exiting the program...")
    sys.exit(0)

def check_size(file_name):
    try:
        df = pd.read_csv(file_name)
        return df.empty
    except EmptyDataError:
        return True
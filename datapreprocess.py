import config
import pandas as pd
from pprint import pprint

def csv_process():

    """
    Takes input as csv and process it and provide required data
    """

    df = pd.read_csv(config.INPUT_FILE, encoding='ISO-8859-1')
    df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)
    df.columns = ["labels","data"]
    df["targets"] = df["labels"].map({'ham':0,'spam':1})
    return df


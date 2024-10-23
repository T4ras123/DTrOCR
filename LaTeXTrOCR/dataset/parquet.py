import parquet
import torch
import os
import PIL
import imagesize
import argparse
import io
import base64
import pandas as pd
from LaTeXTrOCR.utils.utils import path_exists
from LaTeXTrOCR.tokenizer import Tokenizer


ds = pd.DataFrame()

def parquet_to_tensor(path=None) -> pd.DataFrame:
    """Convert parquet file to pandas dataframe 
    and convert bytes to a string of pixel values (H,W,C)

    Args:
        path (str, optional): path to the parquet file. Defaults to None.

    Returns:
        pd.DataFrame: str of pixel values in the "image" column and LaTex in the "text" one
    """
    t = Tokenizer()
    t.load_vocab("../model/dataset/tokenizer.json")
    if path_exists(path):
        df = pd.read_parquet(path)
        print(df)
    else:
        pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=None, help='Path to the parquet file.')
    parser.parse_args()
    
    ds1 = parquet_to_tensor(parser.path)
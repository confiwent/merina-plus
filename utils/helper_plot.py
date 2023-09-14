import os, sys, json
import numpy as np
import pandas as pd


def save_csv(data, file_name):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(file_name, index=False, sep=",")


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        os.system("rm " + folder_path + "/*")


def save_excel(data, path, name="None"):
    if name == "None":
        with pd.ExcelWriter(path) as writer:
            data.to_excel(writer)
    else:
        with pd.ExcelWriter(path) as writer:
            data.to_excel(writer, sheet_name=name)

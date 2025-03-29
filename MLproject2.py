import pandas as pd
from sklearn.model_selection import train_test_split
import os


def stratified_sample_and_save(file_path,name):
    df = pd.read_csv(file_path)
    
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1]   

    X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    train_df = pd.concat([X_train_temp, y_train_temp], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    
    train_df.to_csv(f"{name}_train.csv", index=False)
    val_df.to_csv(f"{name}_val.csv", index=False)
    test_df.to_csv(f"{name}_test.csv", index=False)


csv_files = ['ModifiedDatasetCORR.csv', 'ModifiedDatasetGAINRATIO.csv', 'ModifiedDatasetINFOGAIN.csv', 'ModifiedDatasetONER.csv', 'ModifiedDatasetINDIVIDUAL.csv']
names = ["corr","gainratio","infogain","oneR","indi"]

for file_path,x in zip(csv_files, range(5)):
    stratified_sample_and_save(file_path,names[x])

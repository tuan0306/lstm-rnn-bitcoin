import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def normalize_data(data,scaler=None):
    if scaler is None:
        scaler=MinMaxScaler()
        normalized_data=scaler.fit_transform(data)
    else:
        normalized_data=scaler.transform(data)
    return normalized_data,scaler

def create_sequences(data,lookback=60):
    X=[]
    y=[]
    for i in range(lookback,len(data)):
        X.append(data[i-lookback:i,:])
        y.append(data[i,:])
    X=np.array(X)
    y=np.array(y)
    print(f"Created sequences:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    return X,y

def train_test_split(X,y,train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15):
    n=len(X)
    train_end=int(n*train_ratio)
    valid_end=train_end+int(n*valid_ratio)
    
    X_train,y_train=X[:train_end],y[:train_end]
    X_valid,y_valid=X[train_end:valid_end],y[train_end:valid_end]
    X_test,y_test=X[valid_end:],y[valid_end:]
    
    print("Split sizes:")
    print(" X_train:", X_train.shape, "y_train:", y_train.shape)
    print(" X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
    print(" X_test :", X_test.shape, "y_test :", y_test.shape)

    return X_train,X_valid,X_test,y_train,y_valid,y_test

def save_arrays(X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, folder="data/processed"):
    import os
    os.makedirs(folder,exist_ok=True)
    
    np.save(f"{folder}/X_train.npy", X_train)
    np.save(f"{folder}/X_valid.npy", X_valid)
    np.save(f"{folder}/X_test.npy", X_test)
    np.save(f"{folder}/y_train.npy", y_train)
    np.save(f"{folder}/y_valid.npy", y_valid)
    np.save(f"{folder}/y_test.npy", y_test)
    
    with open(f"{folder}/scaler.pkl","wb") as f:
        pickle.dump(scaler,f)

def load_arrays(folder="data/processed"):
    X_train = np.load(f"{folder}/X_train.npy")
    X_valid = np.load(f"{folder}/X_valid.npy")
    X_test = np.load(f"{folder}/X_test.npy")
    y_train = np.load(f"{folder}/y_train.npy")
    y_valid=np.load(f'{folder}/y_valid.npy')
    y_test = np.load(f"{folder}/y_test.npy")
    
    with open(f"{folder}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler
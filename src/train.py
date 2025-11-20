import argparse
import yaml
import os
from itertools import product
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import joblib


def main(args):
    with open(args.params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
        
    data_train = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    X_train = data_train.drop(columns=["target"])
    y_train = data_train["target"]    
    
    data_test = pd.read_csv(os.path.join(args.data_path, "test.csv"))
    X_test = data_test.drop(columns=["target"])
    y_test = data_test["target"] 
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IRIS RandomForest")

    pdict = {key: value for key, value in params.items() if hasattr(RandomForestClassifier(), key)}

    with mlflow.start_run(run_name="RandomForest__" + "__".join([f"{key}_{value}" for key, value in pdict.items()])):
        model = RandomForestClassifier(
            **pdict
        )
        
        mlflow.log_param("model", "RandomForest")
        for key, value in pdict.items():
            mlflow.log_param(key, value)
            
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")
        
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/", help="Path to data folder, where train.csv and test.csv are located. Relative to root directory. Example: 'data/processed/'")
    parser.add_argument("--params-file", default="params.yaml", help="Path to params yaml file. Relative to root directory. Example: 'params.yaml'")
    
    args = parser.parse_args()
    
    main(args)
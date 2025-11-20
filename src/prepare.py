import argparse
import yaml
import os
from sklearn.model_selection import train_test_split
import pandas as pd


def main(args):
    with open(args.params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
        
    data = pd.read_csv(args.raw_data_file)
    
    data_train, data_test = train_test_split(data,
                                            test_size=params["split_ratio"],
                                            random_state=params["random_state"])
    
    os.makedirs(args.output_path, exist_ok=True)
    
    data_train.to_csv(os.path.join(args.output_path, "train.csv"), index=False)
    data_test.to_csv(os.path.join(args.output_path, "test.csv"), index=False)
        
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-file", default="data/raw/data.csv", help="Path to raw data file. Relative to root directory. Example: 'data/raw/data.csv'")
    parser.add_argument("--output-path", default="data/processed/", help="Path to output data folder. Relative to root directory. Example: 'data/processed/'")
    parser.add_argument("--params-file", default="params.yaml", help="Path to params yaml file. Relative to root directory. Example: 'params.yaml'")
    
    args = parser.parse_args()
    
    main(args)
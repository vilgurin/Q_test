import joblib
import logging
import argparse
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from models import AnalyticalModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(train_path: str, save_path: str) -> None:
    '''
    Main training script
    Args:
        train_path: String path to the input train data.
        save_path: String path to save the trained model.
    '''
    logging.info(f"Loading training data from {train_path}...")

    df_train = pd.read_csv(train_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train['target']
    
    model = AnalyticalModel()
    predictions = model.predict(X_train)
    
    rmse = root_mean_squared_error(y_train, predictions)
    logging.info(f"Validation RMSE: {rmse}")

    logging.info(f"Saving model to {save_path}...")
    joblib.dump(model, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='train.csv', 
                        help='Path to the train data')
    parser.add_argument('--model_output', type=str, default='model.pkl', 
                        help='Path to saved model')
    args = parser.parse_args()
    main(train_path=args.train_data, save_path=args.model_output)

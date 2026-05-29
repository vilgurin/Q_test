import joblib
import logging
import argparse
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main(test_path: str, model_path: str, output_path: str) -> None:
    """
    Main inference function.
    Args:
        test_path: String path to the test data.
        model_path: String path to the trained model.
        output_path: String path to the predictions.
    """

    logging.info(f"Loading test data from {test_path}...") 
    df_test = pd.read_csv(test_path)
    logging.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    logging.info("Generating predictions...")
    predictions = model.predict(df_test)

    logging.info(f"Saving predictions to {output_path}...")
    output = pd.DataFrame({'target': predictions})
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_data', type=str, default='hidden_test.csv', 
                        help='Path to test data')
    parser.add_argument('--model_input', type=str, default='model.pkl', 
                        help='Path to trained model')
    parser.add_argument('--predictions_output', type=str, default='predictions.csv', 
                        help='Path to predictions')
    args = parser.parse_args()

    main(
        test_path=args.test_data, 
        model_path=args.model_input, 
        output_path=args.predictions_output
    )

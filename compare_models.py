from utils.utils import *

def load_processed_data(file_path):
    """
    Load the preprocessed data file.
    
    Args:
        file_path (str): Path to the processed data file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print("Loaded processed DataFrame:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None


def run_xgboost(df):
    """
    Train and evaluate an XGBoost model.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Results metrics
    """
    print("Running XGBoost model...")
    try:
        actual_predicted_targets, feature_importance, final_model, results = train_classifier(
            df, classifier=XGBClassifier(random_state=42)
        )
        results_metrics = get_results_table(results[0], model_name="XGB")
        results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
        return results_metrics
    except Exception as e:
        print(f"Error running XGBoost model: {e}")
        return None


def run_svm(df):
    """
    Train and evaluate an SVM model.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Results metrics
    """
    print("Running SVM model...")
    try:
        actual_predicted_targets, feature_importance, final_model, results = train_classifier(
            df, classifier=SVC(random_state=42, probability=True)
        )
        results_metrics = get_results_table(results[0], model_name="SVM")
        results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
        return results_metrics
    except Exception as e:
        print(f"Error running SVM model: {e}")
        return None


def run_random_forest(df):
    """
    Train and evaluate a Random Forest model.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Results metrics
    """
    print("Running Random Forest model...")
    try:
        actual_predicted_targets, feature_importance, final_model, results = train_classifier(
            df, classifier=RandomForestClassifier(random_state=42)
        )
        results_metrics = get_results_table(results[0], model_name="RF")
        results_metrics["time (s)"] = round(float(results[1]["time"][0]), 2)
        return results_metrics
    except Exception as e:
        print(f"Error running Random Forest model: {e}")
        return None


def run_deepffn(df):
    """
    Train and evaluate a Deep Feed Forward Network model.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Results metrics
    """
    print("Running DeepFFN model...")
    try:
        results, total_time, model = DeepFFN_model(df)
        results_metrics = get_results_table(results=results, model_name="DFFN")
        results_metrics["time (s)"] = round(float(total_time), 2)
        return results_metrics
    except Exception as e:
        print(f"Error running DeepFFN model: {e}")
        return None


def run_ann(df):
    """
    Train and evaluate an Artificial Neural Network model.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Results metrics
    """
    print("Running ANN model...")
    try:
        results, total_time, model = ANN_model(df)
        results_metrics = get_results_table(results=results, model_name="ANN")
        results_metrics["time (s)"] = round(float(total_time), 2)
        return results_metrics
    except Exception as e:
        print(f"Error running ANN model: {e}")
        return None


def save_comparison_results(results_df, output_path):
    """
    Save the model comparison results to a file.
    
    Args:
        results_df (pandas.DataFrame): Comparison results dataframe
        output_path (str): Path to save the results
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the results
        results_df.to_csv(output_path, sep="\t", index=False)
        print(f"Model comparison results saved to {output_path}")
    except Exception as e:
        print(f"Error saving comparison results: {e}")


def combine_results(results_list):
    """
    Combine individual model results into a single dataframe.
    
    Args:
        results_list (list): List of result dataframes
        
    Returns:
        pandas.DataFrame: Combined comparison dataframe
    """
    try:
        # Remove any None values from the list
        valid_results = [r for r in results_list if r is not None]
        
        if not valid_results:
            print("No valid model results to combine")
            return None
        
        # Combine results
        model_comparison = pd.concat(valid_results)
        
        # Reorder columns for better readability
        model_comparison.insert(0, "models", model_comparison["model"])
        model_comparison = model_comparison.drop("model", axis=1)
        
        return model_comparison
    except Exception as e:
        print(f"Error combining model results: {e}")
        return None


def main():
    """
    Main function to parse command line arguments and run the model comparison.
    """
    parser = argparse.ArgumentParser(description='Model Comparison for Cancer Driver Analysis')
    parser.add_argument('--input_data', type=str, required=True, 
                        help='Path to the processed data file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the model comparison results')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['xgb', 'svm', 'rf', 'deepffn', 'ann'],
                        help='List of models to compare (xgb, svm, rf, deepffn, ann)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load the processed data
    df = load_processed_data(args.input_data)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Initialise results list
    results_list = []
    
    # Run selected models
    model_funcs = {
        'xgb': run_xgboost,
        'svm': run_svm,
        'rf': run_random_forest,
        'deepffn': run_deepffn,
        'ann': run_ann
    }
    
    for model in args.models:
        if model.lower() in model_funcs:
            result = model_funcs[model.lower()](df)
            if result is not None:
                results_list.append(result)
        else:
            print(f"Warning: Unknown model '{model}'. Skipping.")
    
    # Combine results
    model_comparison = combine_results(results_list)
    if model_comparison is None:
        print("Failed to combine model results. Exiting.")
        return
    
    # Save comparison results
    save_comparison_results(model_comparison, args.output_file)
    
    print("Model comparison completed successfully.")


if __name__ == "__main__":
    # Run the main function with command line arguments
    main()
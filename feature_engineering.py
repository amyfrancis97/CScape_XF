from utils.utils import *

def load_data_in_chunks(data_path, chunk_size=10000):
    """
    Load large dataset in chunks.
    
    Args:
        data_path (str): Path to the data file
        chunk_size (int): Number of rows to read at once
        
    Returns:
        pandas.DataFrame: Combined dataframe
    """
    chunks = []
    try:
        for chunk in pd.read_csv(data_path, sep="\t", chunksize=chunk_size):
            chunks.append(chunk)
    except Exception as e:
        print(f"Error reading the file: {e}")
    
    return pd.concat(chunks, ignore_index=True) if chunks else None


def preprocess_dataframe(df):
    """
    Preprocess the dataframe by handling string columns and missing values.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Processed dataframe
        list: String columns that were found
    """
    # Identify string columns
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    print("Columns that contain strings:", string_columns)
    
    # Fill missing values
    df = df.fillna(0)
    
    # Exclude unwanted columns (keeping the first three string columns)
    df = df.drop(columns=string_columns[3:], errors='ignore')
    
    return df, string_columns


def run_feature_selection(df, features):
    """
    Trains a model using a subset of features and collects performance metrics.

    Args:
        df (pd.DataFrame): The dataset for training.
        features (list): List of features to include in the model.

    Returns:
        pd.DataFrame: Results table with performance metrics and features.
    """
    try:
        _, _, _, results = train_classifier(df, features=features, classifier=XGBClassifier(random_state=42))
        res = get_results_table(results[0], model_name="XGB")
        # Add elapsed time, number of features, and feature list to the results
        res["elapsed_time (s)"] = results[1]["time"]
        res["number_features"] = len(features)
        res["feature_list"] = [",".join(features)]  # Save features as a single string
        print(res.head())
        return res
    except Exception as e:
        print(f"Error with feature selection for {len(features)} features: {e}")
        return pd.DataFrame()


def ensure_output_directory(output_dir):
    """
    Ensures that the specified output directory exists.

    Args:
        output_dir (str): Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory verified: {output_dir}")


def save_results(df, file_path):
    """
    Saves a DataFrame to a specified file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the file.
    """
    try:
        df.to_csv(file_path, sep="\t", index=False)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results to {file_path}: {e}")


def train_initial_model(df):
    """
    Train an initial model to extract feature importances.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    print("Training initial model to extract feature importances...")
    try:
        _, feature_importance, _, _ = train_classifier(df, classifier=XGBClassifier(random_state=42), feature_importance=True)
        return feature_importance
    except Exception as e:
        print(f"Error during initial model training: {e}")
        return None


def main():
    """
    Main function to parse command line arguments and run the feature selection analysis.
    """
    parser = argparse.ArgumentParser(description='Feature Selection Optimisation for Cancer Driver Analysis')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the input data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Chunk size for reading large files')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of samples to use for feature selection')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Ensure output directory exists
    ensure_output_directory(args.output_dir)
    
    # Load data
    print("Loading data in chunks...")
    df = load_data_in_chunks(args.data_path, args.chunk_size)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    df, _ = preprocess_dataframe(df)
    
    # Sample data for feature importance analysis
    sampled = df.sample(args.sample_size).reset_index(drop=True)
    print(f"Sampled {args.sample_size} rows for feature selection analysis")
    
    # Train initial model to extract feature importances
    feature_importance = train_initial_model(sampled)
    if feature_importance is None:
        print("Failed to train initial model. Exiting.")
        return
    
    # Save detailed feature importances to a separate file
    feature_importance_file = os.path.join(args.output_dir, "feature_importances.txt")
    try:
        feature_importance.to_csv(feature_importance_file, sep="\t", index=False)
        print(f"Feature importances saved to {feature_importance_file}")
    except Exception as e:
        print(f"Error saving detailed feature importances: {e}")
    
    # Generate feature subsets
    importances_to_run = [
        feature_importance["feature"][0:i].tolist() for i in range(1, len(feature_importance))
    ]
    
    # Process each subset of features and collect results
    print("Starting feature selection optimisation...")
    feature_results = [
        run_feature_selection(sampled, feature_list) for feature_list in importances_to_run
    ]
    
    # Combine all result DataFrames
    try:
        feature_res_df = pd.concat(feature_results, ignore_index=True)
    except Exception as e:
        print(f"Error combining feature results: {e}")
        return
    
    # Save results to a file
    feature_results_file = os.path.join(args.output_dir, "number_features_vs_accuracy.txt")
    save_results(feature_res_df, feature_results_file)
    
    # Save detailed feature lists to a separate file
    detailed_features_file = os.path.join(args.output_dir, "detailed_feature_list.txt")
    try:
        feature_res_df[["number_features", "accuracy", "feature_list"]].to_csv(
            detailed_features_file, sep="\t", index=False
        )
        print(f"Detailed feature list saved to {detailed_features_file}")
    except Exception as e:
        print(f"Error saving detailed feature list: {e}")
    
    print("Feature selection analysis completed successfully.")


if __name__ == "__main__":
    # Run the main function with command line arguments
    main()
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


def evaluate_sample_sizes(df, sample_sizes, output_dir):
    """
    Evaluate model performance on different sample sizes.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        sample_sizes (list): List of sample sizes to evaluate
        output_dir (str): Directory to save results
        
    Returns:
        pandas.DataFrame: Results dataframe
    """
    results_list = []
    
    for sample_size in sample_sizes:
        print(f"Processing sample size: {sample_size}")
        try:
            # Subsample the data
            df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Train the classifier
            start_time = time.time()
            _, _, _, results = train_classifier(df_sample, classifier=XGBClassifier(random_state=42))
            elapsed_time = time.time() - start_time
            
            # Get results table
            res = get_results_table(results[0], model_name="XGB")
            res["elapsed_time (s)"] = elapsed_time
            res["number_samples"] = sample_size
            print(res)
            
            # Append results
            results_list.append(res)
        except Exception as e:
            print(f"Error with sample size {sample_size}: {e}")
    
    # Combine results into a DataFrame
    return pd.concat(results_list, ignore_index=True) if results_list else None


def save_and_plot_results(results_df, output_dir):
    """
    Save results to file and generate plots.
    
    Args:
        results_df (pandas.DataFrame): Results dataframe
        output_dir (str): Directory to save results and plots
    """
    # Save results to a file
    results_file = os.path.join(output_dir, "sample_vs_accuracy_processed.txt")
    results_df.to_csv(results_file, sep="\t", index=False)
    print(f"Results saved to {results_file}")
    
    # Plot results
    plot_sample_metrics_smooth_lowess(results_df, output_dir, "number_samples", "Sample Size")
    print(f"Plots saved to {output_dir}")


def main():
    """
    Main function to parse command line arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(description='Cancer Driver Analysis with sample size optimisation')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the input data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Chunk size for reading large files')
    parser.add_argument('--sample_sizes', type=int, nargs='+', 
                        default=[500, 1000, 5000, 10000, 20000, 40000, 80000],
                        help='List of sample sizes to evaluate')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data in chunks...")
    df = load_data_in_chunks(args.data_path, args.chunk_size)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    df, _ = preprocess_dataframe(df)
    
    # Evaluate different sample sizes
    print("Evaluating sample sizes...")
    results_df = evaluate_sample_sizes(df, args.sample_sizes, args.output_dir)
    if results_df is None:
        print("No results generated. Exiting.")
        return
    
    # Save and plot results
    print("Saving and plotting results...")
    save_and_plot_results(results_df, args.output_dir)
    
    print("Analysis completed successfully.")


if __name__ == "__main__":
    # Run the main function with command line arguments
    main()
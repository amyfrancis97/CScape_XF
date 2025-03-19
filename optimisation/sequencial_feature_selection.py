from utils import *

def run_feature_selection(features):
    actual_predicted_targets, feature_importance, final_model, results = train_classifier(df, features = features, classifier = XGBClassifier(random_state=42))
    res = get_results_table(results[0], model_name = "XGB")
    # Add the elapsed time and sample size to the results
    res["elapsed_time (s)"] = results[1]["time"]
    res["number_features"] = len(features)
    print(res.head())
    return res

if __name__ == "__main__":
    # Extract the path to the sampled data and output directory from command-line arguments
    sample_data_path = sys.argv[1]
    outputDir = sys.argv[2]

    df = pd.read_csv(f"{sample_data_path}/sample_cosmic_gnomad43000.txt", sep = "\t")

    actual_predicted_targets, feature_importance, final_model, results = train_classifier(df, classifier = XGBClassifier(random_state=42))
    feature_importance = feature_importance.reset_index(drop = True)
    importances_to_run = [feature_importance["feature"][0:i].tolist() for i in range(1, len(feature_importance))]

    # Process each sampled dataset and collect the results
    feature_res = [run_feature_selection(feature_list) for feature_list in importances_to_run]

    # Concatenate all result DataFrames
    feature_res = pd.concat(feature_res)

    # Save the results to a file
    feature_res.to_csv(f'{outputDir}/number_features_vs_accuracy.txt', sep="\t", index=None)

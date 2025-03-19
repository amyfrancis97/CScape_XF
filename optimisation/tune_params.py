from utils import *

def xgb_grid_search_cv(df, features=None, classifier=XGBClassifier(random_state=42)):
    if features is None:
        X = df.drop(["chrom", "pos", "ref_allele", "alt_allele", "driver_stat", "grouping"], axis=1)
    else:
        X = df[features]
    
    y = df["driver_stat"]
    groups = df["grouping"]

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier=XGBClassifier(random_state=42)
    
    # Define grid
    param_grid = {
        'n_estimators': [50, 100, 200],  # Num. trees
        'max_depth': [3, 5, 7],  # Max. tree depth
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],  # Fraction of samples per tree
        'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features per tree
        'gamma': [0, 0.1, 0.2],  # Minimum loss reduction for splits
        'reg_alpha': [0, 0.1, 1],  # L1 reg.
        'reg_lambda': [1, 1.5, 2],  # L2 reg.
    }

    # Initialise GridSearchCV
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, 
                               scoring='roc_auc', n_jobs=-1, cv=3, verbose=2)
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameter set
    print(f"Best parameters found: {grid_search.best_params_}")

    # Best model
    best_model = grid_search.best_estimator_

    # Predict and evaluate on test set
    y_test_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test set accuracy: {accuracy}")

    best_params = grid_search.best_params_
    best_params_df = pd.DataFrame([best_params])

    # Save to CSV
    best_params_df.to_csv('best_model_params.csv', index=False)

    return grid_search, best_model, accuracy

if __name__ == "__main__":
    # Check if a command-line argument is provided
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the input file as an argument.")
        print("Usage: python tune_params.py <file_path>")
        sys.exit(1)
    
    # Get the file path from command-line argument
    file_path = sys.argv[1]
    
    # Read the DataFrame from the provided file path
    df2 = pd.read_csv(file_path, sep="\t")
    
    # Sample 2000 rows and run the grid search
    xgb_grid_search_cv(df2, features=features)
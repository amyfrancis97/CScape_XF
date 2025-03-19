# CScape_XF

CScape_XF is a bioinformatics toolkit for cancer driver prediction, featuring machine learning models to identify driver mutations from genomic data. This repository contains scripts for data processing, feature selection, model comparison, and hyperparameter optimisation.

## Overview
CScape_XF analyses annotated variant data (from *DrivR-Base*) to identify potential cancer driver mutations using various machine learning algorithms including XGBoost, SVM, Random Forest, and neural network models. The toolkit provides comprehensive functionality for:

* Data preprocessing and feature extraction
* Feature selection optimisation
* Model comparison and evaluation
* Hyperparameter tuning

## Project Structure

```bash
.
CScape_XF/
|-- data/                  # Data storage (gitignored for large files)
|-- models/                # Model implementation modules
|   |-- ANN.py             # Artificial Neural Network implementation
|   |-- DeepFFN.py         # Deep Feed Forward Network implementation
|   |-- metric_results_table.py  # Results formatting utilities
|   |__ train_classifier.py      # Generic classifier training functions
|-- optimisation/          # Hyperparameter optimisation modules
|   |__ tune_params.py     # Parameter tuning functions
|-- outputs/               # Results and output files
|-- plots/                 # Visualisation modules
│   |__ plot_sample_features_vs_metrics.py  # Plotting utilities
|-- scripts/               # Main execution scripts
│   |-- feature_selection.py        # Feature selection script
│   |-- model_comparison.py         # Model comparison script
│   |-- process_data.py             # Data processing script
│   |__  sample_size_optimisation.py # Sample size optimisation script
|-- utils/                 # Utility functions
|-- .gitignore             # Git ignore file
|-- cscape-xf.yml       # Project dependencies
|__ README.md
```
## Installation
Clone the repository:

```bash
git clone https://github.com/amyfrancis97/CScape_XF.git
cd CScape_XF
```

## Dependencies
CScape_XF requires the following dependencies:

```bash
pandas
numpy
scikit-learn
xgboost
tensorflow
matplotlib
seaborn
```

You can install all dependencies using:

```bash
conda env create -f cscape-xf.yml
```

## Sample Size Optimisation
Optimise sample size for model training:

```bash
python scripts/sample_size_optimisation.py --data_path "/path/to/data.csv" --output_dir "/path/to/outputs" --chunk_size 10000 --sample_sizes 500 1000 5000 10000 20000
```

## Feature Selection
Identify and select the most important features:

```bash
python scripts/feature_selection.py --data_path "/path/to/data.csv" --output_dir "/path/to/outputs" --sample_size 1000
```

## Model Comparison
Compare performance of different machine learning models:

```bash
python scripts/model_comparison.py --input_data "/path/to/processed_data.csv" --output_file "/path/to/model_comparison_res.txt" --models xgb svm rf deepffn ann
```

## XGBoost Hyperparameter Optimisation
Optimise XGBoost model parameters:


```bash
python scripts/xgb_optimisation.py --input_data "/path/to/processed_data.csv" --output_dir "/path/to/outputs/" --sample_size 2000 --cv 3
```

## Models
CScape_XF implements and evaluates several machine learning models:

* XGBoost
* Support Vector Machine (SVM)
* Random Forest
* Deep Feed Forward Network (DeepFFN): Custom neural network architecture
* Artificial Neural Network (ANN): Basic neural network implementation

## Results
Results are saved in the specified output directory including:

## Model comparison metrics
* Feature importance rankings
* Hyperparameter optimisation results
* Performance visualisations

## Contributing
Contributions to CScape_XF are welcome. Please feel free to submit a Pull Request.

## Contact
For questions or support, please contact amy.francis@bristol.ac.uk

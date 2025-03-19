# Standard library imports
import os
import sys
import time
import random
from collections import Counter
from typing import *
from itertools import product, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-learn imports
from sklearn import datasets, metrics, model_selection, preprocessing, svm
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, 
    f1_score, mean_squared_error, precision_score, recall_score, 
    roc_auc_score
)
from sklearn.model_selection import (
    GridSearchCV, KFold, LeaveOneGroupOut, RepeatedStratifiedKFold,
    cross_val_predict, train_test_split
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, OneClassSVM, SVC
from sklearn.utils import shuffle

# XGBoost
from xgboost import XGBClassifier, plot_importance

# Imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler

# Custom modules
from config import *
from models.prepare_training_data import prepare_data
from models.train_evaluate import train_and_evaluate_model
from models.metric_results_table import get_results_table
from models.rain_classifier import train_classifier
from models.DeepFFN import *
from models.ANN import *
from optimisation.params import *
from optimisation.sequencial_feature_selection import *
from optimisation.selected_features import *
from plots.plot_sample_features_vs_metrics import plot_sample_metrics_smooth_lowess

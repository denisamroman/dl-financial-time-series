# basic imports
import os, random
import pandas as pd
import numpy as np
import datetime as dt
import ta
from pathlib import Path

# import boruta
from boruta import BorutaPy

# warnings
import warnings
warnings.filterwarnings('ignore')

# plotting & outputs
from pprint import pprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
# Import pyfolio
import pyfolio as pf

from src.helper import *

# sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import f_classif, SelectKBest

# Import VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# import SHAP
import shap
# Import SOM
from minisom import MiniSom
# XGBoost Classifier
from xgboost import XGBClassifier

# metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, auc, roc_curve, plot_roc_curve

# import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

# tensorflow
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 

from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD
from tensorflow.keras.losses import BinaryCrossentropy 
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dropout, Dense, Flatten, SimpleRNN, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import L1L2

# kerastuner
import keras_tuner as kt
from kerastuner import HyperParameters
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband

from multiprocessing import freeze_support
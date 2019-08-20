# Random forest classifier as baseline
# py 3, using "mplace" conda env.

import numpy as np
import pandas as pd
import itertools, os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms
import utils

alg = 'RFC'


def fit_chain(X, y):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    pipeline = Pipeline([
        ('Scale', StandardScaler()),
        (alg, RandomForestClassifier(random_state=0, criterion='gini', class_weight='balanced'))
    ])


if __name__ == '__main__':
    X, y = utils.get_ix_ds()
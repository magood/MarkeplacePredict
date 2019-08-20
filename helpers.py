"""
@author: Matt Good,  portions originally by JTay
"""
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
import matplotlib.pyplot as plt

cv_folds = 5


# Balanced Accuracy is preferred when there is a class imbalance.
def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


scorer = make_scorer(balanced_accuracy)


def plot_learning_curve(ds_name, clf_type, train_size, train_scores, test_scores):
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    plt.figure()
    title = f"Learning Curve: {clf_type}, {ds_name}"
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    train_low = train_scores_mean - train_scores_std
    train_high = train_scores_mean + train_scores_std
    test_low = test_scores_mean - test_scores_std
    test_high = test_scores_mean + test_scores_std

    plt.fill_between(train_size, train_low, train_high, alpha=0.1, color="r")
    plt.fill_between(train_size, test_low, test_high, alpha=0.1, color="g")
    plt.plot(train_size, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_size, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_validation_curve(ds_name, clf_type, df, param_col, param_name):
    # loosely based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    # because of how we get XVALS, this won't work for other algorithms yet.
    plt.figure()
    title = f"Validation Curve: {clf_type}, {ds_name}"
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.grid()

    xvals = df[param_col].values.astype(np.float64)
    train_scores_mean = df['mean_train_score'].values
    train_scores_std = df['std_train_score'].values
    test_scores_mean = df['mean_test_score'].values
    test_scores_std = df['std_test_score'].values

    train_low = train_scores_mean - train_scores_std
    train_high = train_scores_mean + train_scores_std
    test_low = test_scores_mean - test_scores_std
    test_high = test_scores_mean + test_scores_std

    plt.fill_between(xvals, train_low, train_high, alpha=0.1, color="r")
    plt.fill_between(xvals, test_low, test_high, alpha=0.1, color="g")
    plt.semilogx(xvals, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(xvals, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


def plot_validation_curves(ds_name, clf_type, df, param_col, param_name, dataframes, df_names):
    # loosely based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    # because of how we get XVALS, this won't work for other algorithms yet.
    plt.figure()
    title = f"Validation Curve: {clf_type}, {ds_name}"
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.grid()

    colors = ['r', 'g', 'm', 'c', 'y']

    i = 0
    for df, dname in zip(dataframes, df_names):
        xvals = df[param_col].values.astype(np.float64)
        train_scores_mean = df['mean_train_score'].values
        train_scores_std = df['std_train_score'].values
        test_scores_mean = df['mean_test_score'].values
        test_scores_std = df['std_test_score'].values

        train_low = train_scores_mean - train_scores_std
        train_high = train_scores_mean + train_scores_std
        test_low = test_scores_mean - test_scores_std
        test_high = test_scores_mean + test_scores_std

        c1, c2 = colors[2 * i], colors[2 * i + 1]
        plt.fill_between(xvals, train_low, train_high, alpha=0.1, color=c1)
        plt.fill_between(xvals, test_low, test_high, alpha=0.1, color=c2)
        plt.semilogx(xvals, train_scores_mean, 'o-', color=c1, label=f"Train Acc ({dname})")
        plt.plot(xvals, test_scores_mean, 'o-', color=c2, label=f"CV Acc ({dname})")
        i += 1

    plt.legend(loc="best")
    return plt


def basicResults(clfObj, trgX, trgY, tstX, tstY, params, clf_type=None, dataset=None):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise
    cv = ms.GridSearchCV(clfObj, n_jobs=8, param_grid=params, refit=True, verbose=10, cv=cv_folds, scoring=scorer, error_score=0)
    cv.fit(trgX, trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/{}_{}_reg.csv'.format(clf_type, dataset), index=False)

    test_score = cv.score(tstX, tstY)
    # possibly delete file first?  Somehwere, maybe not here, since it's cumulative I think.
    with open('./output/test results.csv', 'a') as f:
        f.write('{},{},{},{}\n'.format(clf_type, dataset, test_score, cv.best_params_))
    # LEARNING CURVES
    train_sizes = np.linspace(0.1, 1.0, 5)  # defaults to: np.linspace(0.1, 1.0, 5)
    train_size, train_scores, test_scores = ms.learning_curve(cv.best_estimator_, trgX, trgY, train_sizes=train_sizes, cv=cv_folds, verbose=10, scoring=scorer, n_jobs=8, shuffle=True)
    curve_train_scores = pd.DataFrame(index=train_size, data=train_scores)
    curve_test_scores = pd.DataFrame(index=train_size, data=test_scores)
    curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_type, dataset))
    curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_type, dataset))
    try:
        p = plot_learning_curve(dataset, clf_type, train_size, train_scores, test_scores)
        print("Learning curve generated, saving...")
        p.savefig('./output/plots/{}_{}_LC.png'.format(clf_type, dataset), bbox_inches='tight')
    except Exception as e:
        print(f"Error generating learning curve plots for {clf_type} {dataset}")
        print(repr(e))
    return cv


# Iteration learning curves    
def iterationLC(clfObj, trgX, trgY, tstX, tstY, params, clf_type=None, dataset=None):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise
    cv = ms.GridSearchCV(clfObj, n_jobs=8, param_grid=params, refit=True, verbose=10, cv=cv_folds, scoring=scorer)
    cv.fit(trgX, trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/ITER_base_{}_{}.csv'.format(clf_type, dataset), index=False)
    d = defaultdict(list)
    name = list(params.keys())[0]
    for value in list(params.values())[0]:      
        d['param_{}'.format(name)].append(value)
        clfObj.set_params(**{name: value})
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(trgX)
        d['train acc'].append(balanced_accuracy(trgY, pred))
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(tstX)
        d['test acc'].append(balanced_accuracy(tstY, pred))
        print(value)
    d = pd.DataFrame(d)
    d.to_csv('./output/ITERtestSET_{}_{}.csv'.format(clf_type, dataset), index=False)
    return cv


def LC_plot(X_train, X_test, y_train, y_test, estimator, clf_type, ds_name, fn_code):
    train_sizes = np.linspace(0.1, 1.0, 5)  # defaults to: np.linspace(0.1, 1.0, 5)
    train_size, train_scores, test_scores = ms.learning_curve(estimator, X_train, y_train, train_sizes=train_sizes, cv=cv_folds, verbose=10, scoring=scorer, n_jobs=8, shuffle=True)
    curve_train_scores = pd.DataFrame(index=train_size, data=train_scores)
    curve_test_scores = pd.DataFrame(index=train_size, data=test_scores)
    curve_train_scores.to_csv('./output/{}_{}_{}_LC_train.csv'.format(clf_type, ds_name, fn_code))
    curve_test_scores.to_csv('./output/{}_{}_{}_LC_test.csv'.format(clf_type, ds_name, fn_code))
    try:
        p = plot_learning_curve(ds_name, clf_type, train_size, train_scores, test_scores)
        print("Learning curve generated, saving...")
        p.savefig('./output/plots/{}_{}_{}_LC.png'.format(clf_type, ds_name, fn_code), bbox_inches='tight')
    except Exception as e:
        print(f"Error generating learning curve plots for {clf_type} {ds_name}")
        print(repr(e))
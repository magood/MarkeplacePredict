# Support Vector Classifier
# py 3, using "mplace" conda env.

import numpy as np
import pandas as pd
import itertools, os
from matplotlib import pyplot as plt
import seaborn as sns
from helpers import  basicResults, iterationLC, scorer
import helpers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import sklearn.model_selection as ms
import utils, json

alg = 'SVM'


def run_for_dataset(X, y, ds, do_C_search=True, do_gamma_search=True, do_final_results=False, final_params=None):
    """Generic run of model selection, training, and analysis for this algorithm on given data."""
    print(f"Starting {alg} analysis on {ds.ds_name}...")
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=0,stratify=y)

    # Gaussian RBF SVC
    # Vary params C, gamma.
    clf_type = f'{alg}_RBF'
    pipe = Pipeline([
        ('Scale', StandardScaler()),
        (alg, SVC(kernel='rbf', class_weight='balanced', random_state=0))
    ])
    rbf_c_params = [10**i for i in range(-4, 6)]
    gamma_default = 1. / X_train.shape[1]  # gamma AUTO (defualt) is 1 / n_features!
    rbf_gamma_params = np.linspace(gamma_default / 50, gamma_default * 7, 30)
    # vary these one at a time so we can chart them more intuitively
    # Start with Gamma.
    if do_gamma_search:
        params = {'SVM__C': [10.], 'SVM__gamma': rbf_gamma_params}
        print("GridSearchCV for RBF gamma starting...")
        np.random.seed(55)
        cv = ms.GridSearchCV(pipe, n_jobs=8, param_grid=params, refit=True, verbose=10, cv=4, scoring=scorer, error_score=0)
        cv.fit(X_train, y_train)
        gamma_search_table = pd.DataFrame(cv.cv_results_)
        gamma_search_table.to_csv(f'./output/{clf_type}_{ds.ds_name}_gamma_reg.csv',index=False)
        p = helpers.plot_validation_curve(ds.ds_name, clf_type, gamma_search_table, 'param_SVM__gamma', 'gamma')
        print("Validation curve Gamma generated, saving...")
        p.savefig(f'./output/plots/{clf_type}_{ds.ds_name}_VC_gamma.png', bbox_inches='tight')
        final_params = cv.best_params_
        with open(f'output/{clf_type}_{ds.ds_name}_gammasearch_finalparams.json', 'w') as fp:
            json.dump(final_params, fp, sort_keys=True, indent=4)
    # Then do C.
    if do_C_search:
        # requires you have done gamma CV already, bootstrapping off best params
        final_params = cv.best_params_
        pipe.set_params(**final_params)
        params = {'SVM__C': rbf_c_params}
        print("GridSearchCV for RBF C starting...")
        np.random.seed(0)
        cv = ms.GridSearchCV(pipe, n_jobs=8, param_grid=params, refit=True, verbose=10, cv=4, scoring=scorer, error_score=0)
        cv.fit(X_train, y_train)
        c_search_table = pd.DataFrame(cv.cv_results_)
        c_search_table.to_csv(f'./output/{clf_type}_{ds.ds_name}_C_reg.csv',index=False)
        p2 = helpers.plot_validation_curve(ds.ds_name, clf_type, c_search_table, 'param_SVM__C', 'C')
        print("Validation curve C generated, saving...")
        p.savefig(f'./output/plots/{clf_type}_{ds.ds_name}_VC_C.png', bbox_inches='tight')
        final_params = cv.best_params_
        with open(f'output/{clf_type}_{ds.ds_name}_Csearch_finalparams.json', 'w') as fp:
            json.dump(final_params, fp, sort_keys=True, indent=4)


    if do_final_results:
        # After running the above, you will narrow down on good parameter ranges for C and gamma.
        # run around this range to for the final analysis process.
        if ds.ds_name == 'ix':
            final_c_range = np.linspace(1, 20, 20)
            # good gamma was: 0.021724138
            final_gamma_range = np.linspace(0.01, 0.2, 10)
        else:
            return
        params = {'SVM__C': final_c_range, 'SVM__gamma': final_gamma_range}
        print(f"Attempting to fine-tune params: {params}")
        np.random.seed(0)
        cv = ms.GridSearchCV(pipe, n_jobs=8, param_grid=params, refit=True, verbose=1, cv=4, scoring=scorer, error_score=0)
        cv.fit(X_train, y_train)
        final_search_table = pd.DataFrame(cv.cv_results_)
        final_search_table.to_csv(f'./output/{clf_type}_{ds.ds_name}_final_reg.csv',index=False)
        final_params = cv.best_params_
        with open(f'output/{clf_type}_{ds.ds_name}_finalparams.json', 'w') as fp:
            json.dump(final_params, fp, sort_keys=True, indent=4)

        # Don't have data learning curves, but I can guess what the answer will be - need more data.

        # Iteration learning curve plots
        # Max iterations turn out to not be a factor in this dataset.
        # print(f"Iteration Learning Curves, using final params: {final_params}")
        # pipe.set_params(**final_params)
        # iter_range = np.linspace(22, 223, 5).astype('int') ** 2
        # iterationLC(pipe, X_train, y_train, X_test, y_test, {'SVM__max_iter': iter_range}, clf_type, ds.ds_name)


if __name__ == '__main__':
    ix = utils.ix
    X, y = ix.get_ds(drop_infrequent_labels=True)
    print(pd.Series(y).value_counts())
    run_for_dataset(X, y, ix, do_final_results=True)

    # Final params found, ~73.3% acc:
    # final_params = {
    #     "SVM__C": 16.0,
    #     "SVM__gamma": 0.052222222222222225
    # }
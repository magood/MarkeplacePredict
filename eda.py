# Exploratory data analysis
# py 3, using "mplace" conda env.

import numpy as np
import pandas as pd
import pickle, itertools, os
from matplotlib import pyplot as plt
import seaborn as sns
from yahoofinancials import YahooFinancials as YF
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import utils

music_fn = 'music.csv'
sp_ticker = '^GSPC'
dow_ticker = '^DJI'
nasdaq_ticker = '^IXIC'
all_tickers = [sp_ticker, dow_ticker, nasdaq_ticker]
nice_markers = ('o', 'v', '^', '<', '>', '1', 's', 'P', '*', '+', 'X', 'D', '_', '|')
rf_outpath = os.path.join('.', 'output', 'RF')
rf_feature_imp_fn = 'rf_feature_imp.csv'


def download(ticker, start_date='2018-02-15', end_date='2015-08-19'):
    yf = YF(ticker)
    # this worked but looks like the dates are reversed?
    # hst = yf.get_historical_price_data('2015-08-19', '2018-02-15', 'daily')
    hst = yf.get_historical_price_data(start_date, end_date, 'daily')
    pricelist = hst[ticker]['prices']
    # example: [{'date': 1439991000, 'high': 2096.169921875, 'low': 2070.530029296875, 'open': 2095.68994140625, 'close': 2079.610107421875, 'volume': 3512920000, 'adjclose': 2079.610107421875, 'formatted_date': '2015-08-19'}]
    df = pd.DataFrame(pricelist)
    df['date'] = pd.to_datetime(df['formatted_date'])
    df.set_index('date', inplace=True)
    df.drop('formatted_date', axis=1, inplace=True)
    return df


def get_ticker_data(ticker, start_date, end_date):
    try:
        df = pd.read_pickle(f"./{ticker}.pkl")
        return df
    except FileNotFoundError:
        df = download(ticker, start_date, end_date)
        df.to_pickle(f"./{ticker}.pkl")
        return df


def augment_financials(df):
    df['swing'] = df['high'] - df['low']
    df['return'] = 0.
    df['return'] = (df['adjclose'] / df['adjclose'].shift(1)) - 1


def get_index_music(ticker):
    sp = get_ticker_data(ticker)
    augment_financials(sp)
    df = pd.DataFrame(index=sp.index)
    mdf = pd.read_csv(music_fn)
    mdf['date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('date', inplace=True)
    mdf.drop('Date', axis=1, inplace=True)
    mdf = mdf[mdf['Music'].isnull() == False]
    df = sp.join(mdf, how='inner')
    return df


def get_music_df():
    mdf = pd.read_csv(music_fn)
    mdf['date'] = pd.to_datetime(mdf['Date'])
    mdf.set_index('date', inplace=True)
    mdf.drop('Date', axis=1, inplace=True)
    mdf = mdf[mdf['Music'].isnull() == False]
    return mdf


def build_index_df(tickers, mindate, maxdate):
    df = None
    for ticker in tickers:
        idx_df = get_ticker_data(ticker, mindate, maxdate)
        augment_financials(idx_df)
        # rename columns with index postfix
        idx_df = idx_df.add_suffix('_' + ticker)
        if df is None:
            df = pd.DataFrame(index=idx_df.index)
        df = idx_df.join(df, how='inner')
    # Now possibly do any inter-index calculations.
    # What is the difference in return across indices from highest to lowest?
    df['max_return'] = df[['return_^GSPC', 'return_^IXIC', 'return_^DJI']].max(axis=1)
    df['min_return'] = df[['return_^GSPC', 'return_^IXIC', 'return_^DJI']].min(axis=1)
    df['return_diff'] = df['max_return'] - df['min_return']
    df = df.dropna()
    return df


def get_all_df(tickers):
    mdf = get_music_df()
    mindate = mdf.index.min().strftime('%Y-%m-%d')
    maxdate = mdf.index.max().strftime('%Y-%m-%d')
    df = build_index_df(tickers, mindate, maxdate)
    df = df.join(mdf, how='inner')
    return df


def scatter_markers(df, xcol, ycol):
    # ensure we have markers for each music selection, looping if necessary.
    music = list(df.Music.unique())
    # ensure we have markers for each music selection, looping if necessary.
    infmarkers = itertools.cycle(nice_markers)
    markers = list(itertools.islice(infmarkers, len(music)))
    for tune, symbol in zip(music, markers):
        df_tune = df[df['Music'] == tune]
        x = df_tune[xcol]
        y = df_tune[ycol]
        plt.scatter(x, y, marker=symbol, label=tune)
    plt.legend()
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title("Marketplace Music Selection")


def rf_feature_imp(X, y, columns):
    np.random.seed(0)
    X = StandardScaler().fit_transform(X)
    dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45]  # grid of dimensions to select
    dims = [d for d in dims if d < X.shape[1]]
    # Always include the actual number of features, too, as a baseline
    if X.shape[1] not in dims:
        dims += [X.shape[1]]

    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0, n_jobs=8)
    fs = rfc.fit(X, y).feature_importances_
    fi = dict(zip(columns, list(fs)))
    ordered_fi = [(k, fi[k]) for k in sorted(fi, key=fi.get, reverse=True)]
    ordered_fi_df = pd.DataFrame(ordered_fi)
    ordered_fi_df.columns = ['feature','importance']
    ordered_fi_df.to_csv(os.path.join(rf_outpath, rf_feature_imp_fn))
    return rfc


def plot_feature_imp(columns):
    plt.close()
    df = pd.read_csv(os.path.join(rf_outpath, rf_feature_imp_fn))
    ax = df.plot.bar(x='feature', y='importance', rot=0, figsize=(40, 10))
    plt.ylabel('Importance')
    plt.title('Feature Importances by Randomized Forest')
    plt.savefig(os.path.join(rf_outpath, 'full_feature_imp.png'), bbox_inches='tight')


def plot_correlations(df):
    plt.close()
    f = plt.figure(figsize=(25, 25))
    df2 = pd.get_dummies(df)
    sns.heatmap(df2.corr(), cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0, linewidths=.5, square=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig(os.path.join(rf_outpath, 'corr_matrix.png'), bbox_inches='tight')
    return f


def drop_useless_dim_prefixes(df, remove_field_prefixes):
    """
        Drops dimensions/columns from the df that do not appear to be useful.
        Provide a list of prefixes for useless columns (remove_field_prefixes).
    """
    droplist = []
    for t in all_tickers:
        for pfx in remove_field_prefixes:
            droplist.append(f'{pfx}_{t}')
    df.drop(droplist, axis=1, inplace=True)
    return df


if __name__ == '__main__':
    df = get_all_df(all_tickers)
    # should get rid of a bunch of stuff we don't think will be predictive before doing a bunch of plots because it's confusing.
    target_field = 'Music'
    columns = df.drop(target_field, 1).columns
    X = df.drop(target_field, 1).copy().values
    y_categorical = df[target_field].copy().values
    le = LabelEncoder()
    le.fit(y_categorical)
    y = le.transform(y_categorical)
    # Scikit learn really wants floats or the scaler will complain
    X = X.astype(np.float64)
    rfc = rf_feature_imp(X, y, columns)
    plot_feature_imp(columns)

    # Look at some correlations here...
    cf = plot_correlations(df)
    
    # Items that are correlated to the music are:
    # Volume, return, swing, return diff, max return, min return.
    # We can see that there are many highly-correlated features, so we can remove many of those.
    # High, low, open, close, adjclose all worthless.
    remove_field_prefixes = ['adjclose', 'close', 'high', 'low', 'open']
    df = drop_useless_dim_prefixes(df, remove_field_prefixes)

    df.to_csv(utils.ix.ds_csv_file_name)

    # print(df.describe())
    # scatter_markers(df, 'return_^GSPC', 'swing_^GSPC')
    # df.groupby('Music').hist()
    # plt.show()
    # some other nice data vis examples: https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
    # Also, conda install -c conda-forge pandas-profiling, then import pandas_profiling, df.profile_report()
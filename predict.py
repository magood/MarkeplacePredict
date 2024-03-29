# Predict for the day using recent market conditions.
# py 3, using "mplace" conda env.
# Matthew Good

import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials as YF
import joblib
import eda, svc, utils
from datetime import datetime, timedelta
import click


def download_ticker(ticker, start_date, end_date):
    yf = YF(ticker)
    hst = yf.get_historical_price_data(start_date, end_date, 'daily')
    pricelist = hst[ticker]['prices']
    # example: [{'date': 1439991000, 'high': 2096.169921875, 'low': 2070.530029296875, 'open': 2095.68994140625, 'close': 2079.610107421875, 'volume': 3512920000, 'adjclose': 2079.610107421875, 'formatted_date': '2015-08-19'}]
    df = pd.DataFrame(pricelist)
    df['date'] = pd.to_datetime(df['formatted_date'])
    df.set_index('date', inplace=True)
    df.drop('formatted_date', axis=1, inplace=True)
    return df


def augment_financials(df):
    """Compute auxiliary statistics such as swing and return."""
    df['swing'] = df['high'] - df['low']
    df['return'] = 0.
    df['return'] = (df['adjclose'] / df['adjclose'].shift(1)) - 1


def build_index_df(tickers, mindate, maxdate):
    df = None
    for ticker in tickers:
        idx_df = download_ticker(ticker, mindate, maxdate)
        augment_financials(idx_df)
        # rename columns with index postfix
        idx_df = idx_df.add_suffix('_' + ticker)
        if df is None:
            df = pd.DataFrame(index=idx_df.index)
        df = idx_df.join(df, how='inner')
    # Now do any inter-index calculations.
    # What is the difference in return across indices from highest to lowest? Etc.
    df['max_return'] = df[['return_^GSPC', 'return_^IXIC', 'return_^DJI']].max(axis=1)
    df['min_return'] = df[['return_^GSPC', 'return_^IXIC', 'return_^DJI']].min(axis=1)
    df['return_diff'] = df['max_return'] - df['min_return']
    df = df.dropna()
    return df


def get_day_dataframe(day):
    """Get index data for the given day in a format suitable for input to a prediction algorithm."""
    # Need to start several days back for gain/loss return calculations, allowing for non-trading days:
    # Future improvements should probably keep local data to eliminate this need.
    mindate = (day - timedelta(days=5)).strftime('%Y-%m-%d')
    targetdate = day.strftime('%Y-%m-%d')
    maxdate = (day + timedelta(days=1)).strftime('%Y-%m-%d')
    df = build_index_df(eda.all_tickers, mindate, maxdate)
    remove_field_prefixes = ['adjclose', 'close', 'high', 'low', 'open']
    df = eda.drop_useless_dim_prefixes(df, remove_field_prefixes)
    try:
        df = df.loc[targetdate]
        return df
    except KeyError as x:
        # Not a trading day.
        return None


def df_to_numpy_array(df):
    """Scikit-learn wants pure numpy arrays usually, and shaped a certain way."""
    X = df.values.astype(np.float64)
    X = X.reshape(1, -1)  # single sample
    assert X.shape == (1, 12), f"Day's data is not in correct shape - expected: (1, 12), actual: {X.shape}"
    return X


def predict_for_day(day):
    ix = utils.ix
    df = get_day_dataframe(day)
    if df is None:
        raise ValueError('Could not get market data for the day, likely the markets are not closed yet.')
    X = df_to_numpy_array(df)
    if X is not None:
        pred = svc.predict(X, ix)
        return pred, df, X


@click.command()
@click.option('--daysago', default=0, help='Offset of day for which to predict music (0=today, 1=yesterday, etc)')
def predict_and_print(daysago):
    """
    Predict the music that will play during the "Numbers" segment on the public radio program Marketplace,
    given that the market results are available for that day.
    """
    prediction_day = datetime.today() - timedelta(days=daysago)
    formatted_day = prediction_day.strftime("%Y-%m-%d")
    pred, df, X = predict_for_day(prediction_day)
    if pred is not None:
        # print(f"Feature vector on {formatted_day}:")
        # print(df)
        print(f"Predicted Music: {pred}")
        with open('pred.txt', 'w') as file:
            file.write(pred)



if __name__ == '__main__':
    predict_and_print()
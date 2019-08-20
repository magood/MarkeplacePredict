import numpy as np
import pandas as pd
import json

def build_df(fn):
    json_data=open(fn).read()
    js_obj = json.loads(json_data)
    data = np.array(js_obj['data'])
    df = pd.DataFrame(js_obj['data'], columns=js_obj['column_names'], index=data[:,0])
    return df

def augment_financials(df):
    df['Swing'] = df['High'] - df['Low']
    df['return'] = (df['Adjusted Close'] / df['Adjusted Close'].shift(1)) - 1

if __name__ == '__main__':
    fn_sp = 'INDEX_SP500.json'
    fn_dow = 'INDEX_DOW.json'
    fn_nasdaq = 'INDEX_NASDAQ.json'
    sp = build_df(fn_sp)
    augment_financials(sp)
    dow = build_df(fn_dow)
    augment_financials(dow)
    ns = build_df(fn_nasdaq)
    augment_financials(ns)

    #build a new dataframe with:
    #average percent change of all three indexes
    #some kind of measure of how closely the indexes agree...  pct diff between highest and lowest?
    #average difference (percent) between high and low for the day of all three indexes
    df = pd.DataFrame(index = sp.index)

    
    #then tack on the results of what music they played (target).
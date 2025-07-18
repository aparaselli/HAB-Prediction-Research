from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
from itertools import permutations
from itertools import combinations
from pyEDM import *
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import time
import os
from time import perf_counter   
from copy import deepcopy
import math
import random
from sklearn.metrics import root_mean_squared_error
from scipy.stats import ttest_ind
import pickle
import ast
import json
import yaml
import argparse
from datetime import datetime, timedelta

import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload



from IPython.display import display, HTML
display(HTML('<style>.container { width:90% !important; }</style>'))

import warnings
warnings.filterwarnings("ignore", 
    message="A worker stopped while some jobs were given to the executor.",
    module="joblib.externals.loky.process_executor")




def get_block(data, num_lags=1, tau=1):
    ''' Get a dataframe with all the possible valid lags of the variables. '''
    
    block = pd.concat([data[var].shift(lag*tau).rename(f'{var}(t-{lag*tau})') for lag in range(num_lags+1) for var in data.columns], axis=1)

    return block
def get_xmap_results_smap(block, target, embeddings, Tp, theta, lib, pred):
    '''Function to do exhaustive search of embeddings.'''
    
    def compute_rho(block, target, embedding, Tp, theta, lib, pred):
        xmap = SMap(dataFrame=block, target=target, columns=embedding, Tp=Tp, theta=theta, embedded=True, lib=lib, pred=pred, noTime=True)
        rho = xmap['predictions'][['Observations', 'Predictions']].corr().iloc[0,1]
        return embedding, xmap['predictions'], rho

    xmap_results = pd.DataFrame(columns=['embedding', 'rho'])
    xmap_results = Parallel(n_jobs=-1)(delayed(compute_rho)(block, target, embedding, Tp, theta, lib, pred) for embedding in embeddings)
    xmap_results = pd.DataFrame(xmap_results, columns=['embedding', 'result', 'rho'])
    xmap_results = xmap_results.sort_values(by='rho', ascending=False).reset_index(drop=True)
    
    return xmap_results

def get_valid_lags_tau(block, target, tau, num_lags, system_variables):
    
    # Get lags of system variables
    system_variable_lags = []
    for var in system_variables:
        var_lags = [f'{var}(t{i})' if i < 0 else f'{var}(t-{i})' for i in range(num_lags * tau, 1)]
        var_lags = var_lags[::tau][:num_lags+1]
        system_variable_lags = system_variable_lags + var_lags
    
    # Remove (t-0) lag of target variable from valid_lags
    valid_lags = [x for x in system_variable_lags if x[-4:-1] != 't-0']

                    
    return valid_lags


def create_single_model(E,theta,target,i_cols,lib, pred,HAB_embed,showPlot=False):
    driver = f'{target}(t-0)'
    cols = i_cols
    pattern = re.compile(r"^CellCountDetection_Limit\(t-\d+\)$")
    cols = [c for c in cols if not pattern.match(c)]
    result = SMap(
        dataFrame = HAB_embed, 
        columns = cols,
        target = driver,
        lib = lib,  # Library from rows 0 to 700
        pred = pred,
        E = E+1,
        theta=theta,
        noTime=True,
        showPlot = showPlot,
        embedded=True,
        ignoreNan = True
    )
    return result

def thresh_bloom_binary_prediction(obs,pred,threshold=8.03199999999999):
    #obs_bloom_95 = np.percentile(obs, 95) #incorrect
    #pred_bloom_95 = np.percentile(pred, 95) #incorrect
    obs_blooms = obs > threshold
    pred_blooms = pred > threshold
    Accuracy = 1 - (obs_blooms ^ pred_blooms).mean()
    True_pos = (obs_blooms & pred_blooms).sum() / obs_blooms.sum()
    False_pos = ((~obs_blooms) & pred_blooms).sum() / (~obs_blooms).sum()
    True_neg = ((~obs_blooms) & (~pred_blooms)).sum() / (~obs_blooms).sum()
    False_neg = (obs_blooms & (~pred_blooms)).sum() / obs_blooms.sum()
    
    return [Accuracy, True_pos, False_pos, True_neg, False_neg]

def bump_lag(col,bump_num): #Created witht the help of ChatGPT, however tested by human
    """
    Change “…(t-i)” ➜ “…(t-(i+bum_num))” in a column name.
    Leaves the name unchanged if no `(t-#)` suffix is present.
    """
    m = re.search(r"\(t-(\d+)\)$", col)      # capture the lag number
    if not m:
        return col                           # nothing to do
    i = int(m.group(1)) + bump_num                  # bump the lag
    return col[: m.start()] + f"(t-{i})"

def create_model(data,params,target,samp,lib,pred,ensemble_sz=300,pred_i_week=0):
    HAB_embed_block = get_block(data,20)
    if pred_i_week != 0:
        HAB_embed_block.columns = [bump_lag(c,pred_i_week) for c in HAB_embed_block.columns]
        HAB_embed_block[f'{target}(t-0)'] = HAB_embed_block[f"{target}(t-{pred_i_week})"].shift(-1*pred_i_week) #Must have the orig t-0 for smap prediction
    parameters = pd.DataFrame(columns=['target', 'columns', 'E', 'theta', 'pred'])
    for i in range(0,min(ensemble_sz*samp+1,params.shape[0]),samp): #CHANGED STEP SIZE
        E = params['E'].iloc[i]
        theta = params['theta'].iloc[i]
        embedding = params['columns'].iloc[i]
        smap_model = create_single_model(E,theta,target,embedding,lib, pred,HAB_embed_block,showPlot=False)
        df = smap_model['predictions']
        #bbp = thresh_bloom_binary_prediction(df['Observations'].iloc[1:-1],df['Predictions'].iloc[1:-1])

        new_row = {'target': target, 'columns': embedding, 'E': E,'theta':theta, 'pred':df['Predictions']}
        parameters.loc[len(parameters)] = new_row
    #print(parameters['columns'].apply(lambda x: x[-3:-1]))
    return parameters



def ensemble_binary_bloom(parameters_df,n=300,p=0.05,samp=1,bloom_thresh=8.013):
    #parameters_df = parameters_df.iloc[0:n*samp:samp]#.sample(n)#CHANGED STEP SIZE
    sum = np.zeros(np.array(parameters_df['pred'].iloc[0][1:]).size)
    for i in range(min(n,parameters_df.shape[0])):
        curr = np.array(parameters_df['pred'].iloc[i][1:]) > bloom_thresh#np.percentile(parameters_df['pred'].iloc[i].iloc[1:],95)#
        sum = sum + curr
    return sum 

'''
@parameters
data (dataframe) - data containing column for target and desired system variables'
params (dataframe) - data containing info for Smap models
target (string) - variable to forecast bloom of

@return
returns forecast for next time step given the dataframe, and number of models which predicted True
'''

def next_forecast(data,params,target,bloom_thresh=7,n=300,p=0.05,samp=1,lib_off=-32,pred_i_week=0):#data,params,target,lib,pred,ensemble_sz=300
    #pred_i_week is to offset how many weeks ur predicting in the future, pred_i_week=0 -> predicting 1 time step out, pred_i_week=1 -> predicting 2 timesteps out, etc
    lib = '1 ' + str(data.shape[0] + lib_off)#lib = '1 ' + str(data.shape[0] + lib_off) 
    pred = '' + str(data.shape[0] + lib_off + 1) + ' ' + str(data.shape[0]) #pred = '' + str(data.shape[0] + lib_off + 1) + ' ' + str(data.shape[0])
    parameters = create_model(data,params,target,samp,lib,pred,ensemble_sz=n,pred_i_week=pred_i_week)
    preds = ensemble_binary_bloom(parameters,n=n,p=p,samp=samp,bloom_thresh=bloom_thresh)
    return preds> (n*p), preds


def clean_data(data, path=True):
    if path:
        paper_data = pd.read_csv(data)
    else:
        paper_data=data
    #paper_data = adjust_live_data(df)
    paper_data = paper_data.set_index('time')
    corr_date = paper_data['DATE']
    paper_data['Time'] = paper_data.index.astype(int)
    paper_data['Avg_Chloro'] #= paper_data['Avg_Chloro'].apply(np.log1p) #LOG AMPUTATION
    #IMPUTE HAB DATA
    #Build basic linear regression model as sanity check
    # Custom impute missing values with the average of the value in front and behind of it 
    class ForwardBackwardImputer(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_filled_forward = X.fillna(method='ffill').fillna(method='bfill')
            X_filled_backward = X.fillna(method='bfill').fillna(method='ffill')

            return (X_filled_forward + X_filled_backward) / 2


    Imputer = ForwardBackwardImputer()
    paper_data = paper_data.apply(pd.to_numeric, errors='coerce')
    Imputer.fit(paper_data)
    paper_data = Imputer.transform(paper_data)#COMMENT OUT IF DONT WANT MEAN MPUTE
    paper_data['DATE'] = corr_date
    return paper_data

def get_live_data(service_json_path):
    FILE_ID = "1bHrmzw5W5cz8DE-ggEwA3WSEDuszG72UUvKQRBBnLPA"     # your sheet
    SERVICE_JSON = service_json_path

    # --- auth ---
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_JSON,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    drive = build("drive", "v3", credentials=creds)

    # --- export Google Sheet -> CSV ---
    request = drive.files().export_media(
        fileId=FILE_ID,
        mimeType="text/csv",
        # **required for Shared Drives** :contentReference[oaicite:0]{index=0}
    )

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()          # progress in chunks

    fh.seek(0)                     # rewind the buffer
    df = pd.read_csv(fh)           # straight into pandas
    return df

def adjust_live_data(past_data_path, target,json_key_path):  
    live_df = get_live_data(json_key_path)
    # Get the total counts 
    cells_cols = live_df.filter(regex=r"\(cells/L\)$").columns
    live_df[cells_cols] = live_df[cells_cols].apply(pd.to_numeric, errors="coerce")

    diatom_cols = [
        "Pseudo-nitzschia delicatissima group (cells/L)",
        "Pseudo-nitzschia seriata group (cells/L)",
        "Other Diatoms (cells/L)",
    ]

    dinoflag_cols = [
        "Akashiwo sanguinea  (cells/L)",
        "Alexandrium spp. (cells/L)",
        "Ceratium spp.(cells/L)",
        "Cochlodinium/Margalefidinium spp. (cells/L)",
        "Dinophysis spp. (cells/L)",
        "Gymnodinium spp. (cells/L)",
        "Lingulodinium polyedra (cells/L)",
        "Prorocentrum spp. (cells/L)",
        "Other Dinoflagellates (cells/L)",
    ]


    live_df["Total_Diatoms"]        = live_df[diatom_cols].sum(axis=1, min_count=1)
    live_df["Total_Dinoflagellates"] = live_df[dinoflag_cols].sum(axis=1, min_count=1)
    live_df["Total_Prorocentrum_spp"] = pd.to_numeric(
        live_df["Prorocentrum spp. (cells/L)"], errors="coerce"
    )
    live_df["Total_Cochlodinium_spp"]  = pd.to_numeric(
        live_df["Cochlodinium/Margalefidinium spp. (cells/L)"], errors="coerce"
    )
    live_df["Total_Tripos"] = pd.to_numeric(
        live_df["Ceratium spp.(cells/L)"],   
        errors="coerce"                 
    )


    live_df.columns = (
        live_df.columns
        .str.replace(r"\s*\(.*?\)", "", regex=True)   
        .str.strip()                                 
        .str.replace(" ", "_")                       
    )


    live_df.columns = live_df.columns.str.replace(r"__+", "_", regex=True)
    live_df['time'] = live_df.index
    #Replace live data missing values with carter dataset
    paper_data = clean_data(past_data_path)
    # Convert the yyyymmdd integers to the same yyyy-mm-dd string format
    live_df["SampleID"] = (
        live_df["SampleID"]
            .astype(str)                       
            .str.zfill(8)                      
            .pipe(pd.to_datetime, format="%Y%m%d") 
            .dt.strftime("%Y-%m-%d")           # → "2025-04-28"
    )
    paper_data['DATE'] = pd.to_datetime(paper_data['DATE'])

    clean_dates = (
        live_df["SampleID"]
        .astype(str)                       # 2008-06-30 → "2008-06-30"
        .str.replace(r"\D", "", regex=True)  # drop non-digits → "20080630"
        .str.zfill(8)                      # make sure it's 8 chars
    )
    
    live_df["DATE"] = (
        pd.to_datetime(clean_dates, format="%Y%m%d")  # Timestamp dtype
        .dt.strftime("%Y-%m-%d")                      # "YYYY-MM-DD" strings
    )



    target_lookup = paper_data.set_index("DATE")[target]
    live_df[target] = live_df[target].fillna(live_df["DATE"].map(target_lookup))
    live_df = clean_data(live_df,path=False)

    # Add any missing rows 
    paper_data.reset_index()
    KEYS = ["DATE"]        # <- change to your key names

    # --- 2.  Pick out the "new" columns that df1 is missing  ---------------------
    new_cols =paper_data.columns.difference(live_df.columns)    # pandas Index object

    # --- 3.  (Optional but wise) be sure df2 has unique rows per key  ------------
    paper_data_clean = paper_data.drop_duplicates(subset=KEYS)


    for df in (live_df, paper_data_clean):
        df["DATE"] = (
            pd.to_datetime(df["DATE"], errors="coerce")   # strings ➜ datetime64[ns]
            .dt.normalize()                             # drop the time part (midnight) – optional
        )
    # --- 4.  Merge the new columns onto df1  -------------------------------------
    live_df = ( #df1 = live_df
        live_df.merge(
            paper_data_clean[KEYS + new_cols.tolist()],      # only keys + new cols
            on=KEYS,                                  # match on the keys
            how="left"                                # keep all rows in df1
        )
    )
    return live_df

def proxy_confidence_of_pred(num_pos, n, p):
    #num_pos is the number of models that predict their is a bloom
    #Returns a decimal of the likelihood
    if n*p < num_pos: #Ensemble Predicts there is a bloom
        CI = ((num_pos-(n*p))/n-(n*p))
        CI = min(.9,max(.1,CI))
        return CI
    else: #Ensemble predicts no bloom
        CI = 1-(num_pos/(n*p))
        CI = min(.9,max(.1,CI))
        return CI


def str_to_list(s):
    s = s.replace('nan', 'null')  # Replace 'nan' with 'null' for JSON compatibility
    lst = json.loads(s)  # Convert string to list
    lst = [np.nan if x is None else x for x in lst]  # Replace None with np.nan
    return lst

def process_parameters(path):

    parameters = pd.read_csv(path) 
    parameters['pred'] = parameters['pred'].apply(str_to_list)
    parameters['columns'] = parameters['columns'].apply(ast.literal_eval)
    parameters.sort_values(by='rho',ascending=False)
    return parameters


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def eval_ensemble(obs_blooms, pred_blooms):
    Accuracy = 1 - (obs_blooms ^ pred_blooms).mean()
    True_pos = (obs_blooms & pred_blooms).sum() / obs_blooms.sum()
    False_pos = ((~obs_blooms) & pred_blooms).sum() / (~obs_blooms).sum()
    True_neg = ((~obs_blooms) & (~pred_blooms)).sum() / (~obs_blooms).sum()
    False_neg = (obs_blooms & (~pred_blooms)).sum() / obs_blooms.sum()
    
    return [Accuracy, True_pos, False_pos, True_neg, False_neg]


def main():
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_yaml(args.config)
    print("YAML Contents:")
    for key, value in config.items():
        if key != "json_key_path":
            print(f"{key}: {value}")
    t0 = perf_counter()
    #For one week out prediction
    data = adjust_live_data(config['data_path'],config['target'],config['json_key_path'])   
    parameters = process_parameters(config['parameters_path_1wk'])
    forecast, num_models = next_forecast(data,parameters,config['target'],bloom_thresh=config['bloom_thresh'],n=config['n'],p=config['p'],samp=config['samp'])
    #output to JSON
    elapsed = perf_counter() - t0 
    print(f"[1-week model] runtime: {elapsed:.2f} seconds")
    #Calculate next week 
    #dt = datetime.strptime(data['DATE'].iloc[-1], "%Y-%m-%d")   
    dt = data['DATE'].iloc[-1].to_pydatetime() 

    next_week_dt = dt + timedelta(days=7)               
    next_week_str = next_week_dt.strftime("%Y-%m-%d")    
    weekday_name  = next_week_dt.strftime("%A")          

    #Gather results
    print(f'Bloom prediction 1wk: {forecast[-1]}')
    print(f"Confidence for 1wk pred: {proxy_confidence_of_pred(num_models[-1], config['n'],config['p'])*100}")
    print(f'Num of models which predict bloom 1wk: {num_models[-1]}')
    if forecast[-1]:
        forecast_pred_1wk = "Likely"
    else:
        forecast_pred_1wk = "Unlikely"
    out_JSON_data = []
    #Append 1 week pred 
    out_JSON_data.append({"Bloom_type": "Bioluminescent", "day": weekday_name,   "Date": next_week_str, "CI": (proxy_confidence_of_pred(num_models[-1], config['n'],config['p'])*100), "Likeliness": forecast_pred_1wk})

    #For two week out prediction
    t0 = perf_counter()
    parameters = process_parameters(config['parameters_path_2wk'])
    forecast, num_models = next_forecast(data,parameters,config['target'],bloom_thresh=config['bloom_thresh'],n=config['n'],p=config['p'],samp=config['samp'],pred_i_week=1)
   
    elapsed = perf_counter() - t0 
    print(f"[2-week model] runtime: {elapsed:.2f} seconds")

    #Calculate two week 
    #dt = datetime.strptime(data['DATE'].iloc[-1], "%Y-%m-%d") 
    dt = data['DATE'].iloc[-1].to_pydatetime()   

    next_week_dt = dt + timedelta(days=14)               
    next_week_str = next_week_dt.strftime("%Y-%m-%d")    
    weekday_name  = next_week_dt.strftime("%A")          

    #Gather results
    print(f'Bloom prediction 2wk: {forecast[-1]}')
    print(f"Confidence for 2wk pred: {proxy_confidence_of_pred(num_models[-1], config['n'],config['p'])*100}")
    print(f'Num of models which predict bloom 2wk: {num_models[-1]}')
    if forecast[-1]:
        forecast_pred_2wk = "Likely"
    else:
        forecast_pred_2wk = "Unlikely"
    #Append 2 week pred 
    out_JSON_data.append({"Bloom_type": "Bioluminescent", "day": weekday_name,   "Date": next_week_str, "CI": (proxy_confidence_of_pred(num_models[-1], config['n'],config['p'])*100), "Likeliness": forecast_pred_2wk})


    #For 3 week out predicting 
    #For two week out prediction
    t0 = perf_counter()
    parameters = process_parameters(config['parameters_path_3wk'])
    forecast, num_models = next_forecast(data,parameters,config['target'],bloom_thresh=config['bloom_thresh'],n=config['n'],p=config['p'],samp=config['samp'],pred_i_week=2)
    #output to JSON
    elapsed = perf_counter() - t0 
    print(f"[3-week model] runtime: {elapsed:.2f} seconds")

    #Calculate three week 
    #dt = datetime.strptime(data['DATE'].iloc[-1], "%Y-%m-%d")   
    dt = data['DATE'].iloc[-1].to_pydatetime() 

    next_week_dt = dt + timedelta(days=21)               
    next_week_str = next_week_dt.strftime("%Y-%m-%d")    
    weekday_name  = next_week_dt.strftime("%A")          

    #Gather results
    print(f'Bloom prediction 3wk: {forecast[-1]}')
    print(f"Confidence for 3wk pred: {proxy_confidence_of_pred(num_models[-1], config['n'],config['p'])*100}")
    print(f'Num of models which predict bloom 3wk: {num_models[-1]}')
    if forecast[-1]:
        forecast_pred_3wk = "Likely"
    else:
        forecast_pred_3wk = "Unlikely"
    #Append 2 week pred 
    out_JSON_data.append({"Bloom_type": "Bioluminescent", "day": weekday_name,   "Date": next_week_str, "CI": (proxy_confidence_of_pred(num_models[-1], config['n'],config['p'])*100), "Likeliness": forecast_pred_3wk})

    #SAVE FILE TO bloom_forecast.json
    out_file = Path("outputs") / "bloom_forecast.json"   # e.g. ./outputs/bloom_forecast.json
    out_file.parent.mkdir(parents=True, exist_ok=True)   # make the folder if it’s missing

    # 3 ️⃣  write (or overwrite) the file
    with out_file.open("w") as f:
        json.dump(out_JSON_data, f, indent=2)  
    '''
    testing = False
    if testing:
        start_index = 515
        observations = np.array(data[config['target']].iloc[start_index:]) > config['bloom_thresh']
        print(observations)
        forecasts = []
        for i in range(start_index ,data.shape[0]):
            temp_data = data[:i] #will forecast data point @ start_index
            forecast, num_models = next_forecast(temp_data,parameters,config['target'],bloom_thresh=config['bloom_thresh'],n=config['n'],p=config['p'],samp=config['samp'])
            forecasts.append(forecast[-1])
            print(f'\n number of models predicted bloom: {num_models[-1]}')
            print(f'Chloro level: {np.array(data[config['target']].iloc[start_index:])[i-start_index]}')
            if forecast[-1] == observations[i-start_index]:
                print('Correct')
            else:
                print('Incorrect')
        results = eval_ensemble(observations, np.array(forecasts))
        print(f'Accuracy: {results[0]}')
        print(f'True Positive: {results[1]}')
        print(f'False Positive {results[2]}')
        print(f'True Negative: {results[3]}')
        print(f'False Negative: {results[4]}')
        print(forecasts)
        '''

if __name__ == "__main__":
    main()
    

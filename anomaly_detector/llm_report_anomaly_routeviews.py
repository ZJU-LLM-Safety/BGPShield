#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
# from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from pandas.errors import EmptyDataError
from utils import approx_knee_point, event_aggregate, approx_knee_point_continuous
import json
import pandas as pd
import numpy as np
import csv
import click
from tqdm import tqdm

repo_dir = Path(__file__).resolve().parent.parent


def metric_threshold(df, metric_col):
    """
    用来计算某列数据的“异常值”阈值。
    """
    values = df[metric_col] 
    trimmed = values[values <= values.quantile(0.997)]
    mu = np.mean(trimmed)   
    sigma = np.std(trimmed)  
    metric_th = mu+4*sigma  

    return metric_th

def knee_metric_threshold(df, metric_col):
    values = df[metric_col].values
    th, cf = approx_knee_point(values)

    return th

def forwarder_threshold(df, event_key):
    route_changes = tuple(df.groupby(event_key))    
    forwarder_num = [len(j["forwarder"].unique()) for _, j in route_changes]    
    forwarder_th, cdf = approx_knee_point_continuous(forwarder_num)    


    return forwarder_th

def window(df0, df1, # df0 for reference, df1 for detection
        metric="diff", event_key=["prefix1", "prefix2"],
        dedup_index=["prefix1", "prefix2", "forwarder", "diff_path_1", "diff_path_2"]):

    if dedup_index is not None:
        df0 = df0.drop_duplicates(dedup_index, keep="first", inplace=False, ignore_index=True)


    with pd.option_context("mode.use_inf_as_na", True):
        df0 = df0.dropna(how="any")
        # df1 = df1.dropna(how="any")

    # print(f"df0: {df0}")
    # print(f"df1: {df1}")
    # print(f"----------------------\n")
    
    metric_th = metric_threshold(df0, metric)
    # metric_th = knee_metric_threshold(df0, metric)
    forwarder_th = forwarder_threshold(df0, event_key)

    events = {}
    for key, ev in tuple(df1.groupby(event_key)):
        if len(ev["forwarder"].unique()) <= forwarder_th: continue  
        
        ev_sig = ev.sort_values(metric, ascending=False).drop_duplicates("forwarder")   

        ev_anomaly = ev_sig.loc[ev_sig[metric]>metric_th]   
        if ev_anomaly.shape[0] <= forwarder_th: continue    

        events[key] = ev_anomaly    

    if events:
        _, df = event_aggregate(events)
        n_alarms = len(df['group_id'].unique())
    else:
        df = None
        n_alarms = 0

    info = dict(
        metric=metric,
        event_key=event_key,
        metric_th=float(metric_th),
        forwarder_th=int(forwarder_th),
        n_raw_events=len(events),
        n_alarms=n_alarms,
    )

    return info, df

@click.command()
@click.option("--collector", "-c", type=str, default="all_collectors", help="the name of RouteView collector to detect anomalies")
@click.option("--year", "-y", type=int, required=True, help="the year of the route changes monitored, e.g., 2024")
@click.option("--month", "-m", type=int, required=True, help="the month of the route changes monitored, e.g., 8")
@click.option("--day", "-d", type=int, default=None, help="the certain day of the route changes monitored, e.g., 1")
@click.option("--hour", "-H", type=int, default=12)
@click.option("--minute", "-M", type=int, default=0)
@click.option("--model", "-M", type=int, default=0, help="id of LLM Model")
@click.option("--dimension", "-d", type=int, default=0, help="Whether to use dimensionality reduction, 0 for NonReduce, Positive Number for dimension of reduction")
def report_alarm(collector, year, month, day, hour, minute, model, dimension):
    model_list = [
        "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/", 
        "/hub/huggingface/models/Qwen/Qwen3-1.7B-Base/", 
        "/hub/huggingface/models/BAAI/bge-m3/",
        "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
        ]
    
    model_path = model_list[int(model)]

    bge = True if model_path == "/hub/huggingface/models/BAAI/bge-m3/" else False
    reduce = True if dimension > 0 else False

    model_name = model_path.split("/")[-2]
    print(f"Model: {model_name}")

    if day is not None:
        target_date = f"{year}{month:02d}{day:02d}{hour:02d}{minute:02d}"
    else:
        target_date = f"{year}{month:02d}"

    collector_result_dir = repo_dir/"routing_monitor"/"detection_result"/collector
    route_change_dir = collector_result_dir/f"route_change_{target_date}"
    # route_change_dir = collector_result_dir/f"route_change"
    
    print("use llm model")
    llm_metric_dir = collector_result_dir/model_name/f"llm_metric_{target_date}_{dimension}"
    reported_alarm_dir = collector_result_dir/model_name/"Llm_reported_alarms"/f"{target_date}_{dimension}"
    if not reduce:
        llm_metric_dir = collector_result_dir/model_name/"NonReduce"/f"llm_metric_{target_date}"
        reported_alarm_dir = collector_result_dir/model_name/"NonReduce"/"Llm_reported_alarms"/f"{target_date}"
    
    reported_alarm_dir.mkdir(parents=True, exist_ok=True)
    if Path(reported_alarm_dir/f"info_{target_date}.json").exists():
        print(f"Reported alarms for {target_date} already exist.")
        return

    def preprocessor(df):
        if 'diff' in df.columns:
            df["diff_balance"] = df["diff"] / (df["aligned_count"] + 1e-8)  # aligned_count
        return df
    
    def load_data(year, month, preprocessor=lambda df: df):
        target_date = f"{year}{month:02d}"
        route_change_files = sorted(route_change_dir.glob(f"{target_date}*.csv"), key=lambda f: f.stat().st_mtime)
        embds_metric_files = sorted(llm_metric_dir.glob(f"{target_date}*.bm.csv"), key=lambda f: f.stat().st_mtime)
        datetimes = [i.stem.replace(".","")[:-2] for i in route_change_files]  

        bulk_datetimes, bulk_indices = np.unique(datetimes, return_index=True)
        bulk_ranges = zip(bulk_indices, bulk_indices[1:].tolist()+[len(datetimes)])

        def load_one_bulk(i,j):
            valid_dfs = []

            for f in embds_metric_files[i:j]:
                try:
                    df = pd.read_csv(f, quotechar='"')
                    if df.empty:
                        continue
                    valid_dfs.append(df)
                except EmptyDataError:
                    continue
                except Exception as e:
                    print(e)
                    continue

            if not valid_dfs:
                return pd.DataFrame()
            
            return pd.concat(valid_dfs, ignore_index=True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            raw_bulks = list(executor.map(
                        lambda x: preprocessor(load_one_bulk(*x)), bulk_ranges))

        filtered = [(dt, df) for dt, df in zip(bulk_datetimes, raw_bulks) if not df.empty]
        if filtered:
            bulk_datetimes, bulks = zip(*filtered)
            return list(bulk_datetimes), list(bulks)
        else:
            return [], []
        # return bulk_datetimes, bulks

    datetimes, bulks = load_data(year, month, preprocessor)
    indices = np.arange(len(bulks))
    infos = []
    total_num = 0
    indice = list(zip(indices[:-1], indices[1:]))
    for i, j in tqdm(indice, desc="Detecting anomalies", total=len(indice)-1):
        save_path = reported_alarm_dir/f"{datetimes[i]}_{datetimes[j]}.alarms.csv"
        info = dict(d0=datetimes[i], d1=datetimes[j])
        n = 1
        refer_bulks = bulks[max(0, i-n):i+1]
        refer_df = pd.concat(refer_bulks, ignore_index=True)
        # _info, df = window(bulks[i], bulks[j], metric="diff_balance")
        _info, df = window(refer_df, bulks[j], metric="diff_balance")
        info.update(**_info)

        if df is None:
            info.update(save_path=None)
        else:
            total_num += df.shape[0]
            
            df["detect_time"] = datetimes[j]
            df.to_csv(save_path, index=False, mode="w", quoting=csv.QUOTE_NONNUMERIC)
            info.update(save_path=str(save_path))

        infos.append(info)
        

    print(f"Total alarms detected: {total_num}")
    print(f"Reported alarms for {target_date} saved to {reported_alarm_dir}")
    json.dump(infos, open(reported_alarm_dir/f"info_{target_date}.json", "w"), indent=2)

if __name__ == "__main__":
    report_alarm()
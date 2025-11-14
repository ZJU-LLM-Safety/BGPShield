#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm, trange
import csv
from pandas.errors import EmptyDataError
from utils import load_embs_distance_optim

@click.command()
@click.option("--collector", "-c", type=str, default="all_collectors", help="the name of RouteView collector that the route changes to evaluate are from")
@click.option("--year", "-y", type=int, required=True, help="the year of the route changes monitored, e.g., 2024")
@click.option("--month", "-m", type=int, required=True, help="the month of the route changes monitored, e.g., 8")
@click.option("--day", "-d", type=int, default=None, help="the certain day of the route changes monitored, e.g., 1")
@click.option("--hour", "-H", type=int, default=12)
@click.option("--minute", "-M", type=int, default=0)
@click.option("--model", "-M", type=int, default=0, help="id of LLM Model")
@click.option("--dimension", "-d", type=int, default=0, help="Whether to use dimensionality reduction, 0 for NonReduce, Positive Number for dimension of reduction")
@click.option("--epoches", "-e", type=int, default=150, help="the number of epoches for RNN")
def evaluate_monthly_for(collector, year, month, day, hour, minute, model, dimension, epoches):
    model_list = [
        "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/", 
        "/hub/huggingface/models/Qwen/Qwen3-1.7B-Base/", 
        "/hub/huggingface/models/BAAI/bge-m3/",
        # "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
        ]
    
    model_path = model_list[int(model)]

    bge = True if model_path == "/hub/huggingface/models/BAAI/bge-m3/" else False
    reduce = True if dimension > 0 else False

    model_name = model_path.split("/")[-2]
    print(f"Model: {model_name}")

    repo_dir = Path(__file__).resolve().parent.parent
    collector_result_dir = repo_dir/"routing_monitor"/"detection_result"/ collector

    if day is not None:
        target_date = f"{year}{month:02d}{day:02d}{hour:02d}{minute:02d}"
    else:
        target_date = f"{year}{month:02d}"
    
    # 输入数据
    # route_change_dir = collector_result_dir / f"route_change"
    route_change_dir = collector_result_dir / f"route_change_{target_date}"

    print("use llm model")
    llm_metric_dir = collector_result_dir/model_name/f"llm_metric_{target_date}_{dimension}"

    if not reduce:
        llm_metric_dir = collector_result_dir/model_name/"NonReduce"/f"llm_metric_{target_date}"

    llm_metric_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving to {llm_metric_dir}")

    llm_emb_path = Path(__file__).resolve().parent.parent/"BGPShield"/"llMmodels"/"moreprompt"
    # llm_emb_path = Path(__file__).resolve().parent.parent/"BGPShield"/"llMmodels"/"newprompt"

    comp_date = int(f"{year}{month:02}01")
    if reduce:
        model_name = f"{model_name}/L2/{year}{month:02}01.as-rel{'2.' if comp_date > 20151201 else '.'}{epoches}.10.{dimension}"

    llm_emb_dir = llm_emb_path/f"mean/iterative_as_info/{year}{month:02d}01/{model_name}"

    print(f"Loading {llm_emb_dir}")
    emb_d, dtw_d, path_d, emb = load_embs_distance_optim(llm_emb_dir, bge, return_emb=True)

    def dtw_d_only_exist(s, t):
        return dtw_d([i for i in s if i in emb], [j for j in t if j in emb])

    # for i in route_change_dir.glob(f"{year}{month:02d}*.csv"):
    sorted_files = sorted(
        route_change_dir.glob(f"{int(year)}{int(month):02d}*.csv"),
        key=lambda f: f.stat().st_mtime
    )
    for i in tqdm(sorted_files, 
                  desc="Evaluating route changes", 
                  total=len(list(sorted_files))):
        beam_metric_file = llm_metric_dir/f"{i.stem}.bm.csv"
        if beam_metric_file.exists(): continue

        try:
            df = pd.read_csv(i, quotechar='"')
        except EmptyDataError:
            # 如果原始 CSV 文件为空，跳过
            continue
        except Exception as e:
            # 捕获其他读取错误，建议至少打印警告
            print(f"Error reading {i}: {e}")
            continue
        
        # 从 csv 文件中提取路径
        path1 = [s.split(" ") for s in df["path1"].values]
        path2 = [t.split(" ") for t in df["path2"].values]

        diff = [dtw_d(s, t) for s, t in zip(path1, path2)]
        diff_only_exist = [dtw_d_only_exist(s, t) for s, t in zip(path1, path2)]

        processed_path1 = [d[1] for d in diff]
        processed_path2 = [d[2] for d in diff]

        metrics = pd.DataFrame.from_dict({
            # 添加标识
            "timestamp": df["timestamp"],
            "vantage_point": df["vantage_point"],
            "forwarder": df["forwarder"],
            "prefix1": df["prefix1"],
            "prefix2": df["prefix2"],
            "path1": df["path1"],
            "path2": df["path2"],
            # 计算数据
            "diff": [d[0] for d in diff],    # [dtw_d(s,t) for s,t in zip(path1, path2)],  
            "diff_path_1": processed_path1,    
            "diff_path_2": processed_path2,    
            "diff_only_exist": [d[0] for d in diff_only_exist], #[dtw_d_only_exist(s,t) for s,t in zip(path1, path2)],    
            "diff_only_exist_path_1": [d[1] for d in diff_only_exist],
            "diff_only_exist_path_2": [d[2] for d in diff_only_exist],
            "path_d1": [path_d(i) for i in processed_path1], # [path_d(i) for i in path1],  # 计算路径的嵌入长度
            "path_d2": [path_d(i) for i in processed_path2], # [path_d(i) for i in path2],  # 计算路径的嵌入长度
            "path_l1": [len(i) for i in processed_path1], # [len(i) for i in path1],     # 路径的长度
            "path_l2": [len(i) for i in processed_path2] ,# [len(i) for i in path2],     # 路径的长度
            "head_tail_d1": [emb_d(i[0], i[-1])[0] for i in processed_path1] , # [emb_d(i[0], i[-1]) for i in path1],    
            "head_tail_d2": [emb_d(i[0], i[-1])[0] for i in processed_path2] ,# [emb_d(i[0], i[-1]) for i in path2],   
            "aligned_count": [d[3] for d in diff] ,
        })
        
        metrics.to_csv(beam_metric_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == "__main__":
    evaluate_monthly_for()

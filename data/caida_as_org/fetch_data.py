#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import json
import subprocess
import click
import re

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = SCRIPT_DIR/"fetched_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_archive_list(refresh=False):
    cache_path = CACHE_DIR/f"time2url"
    if cache_path.exists() and not refresh:
        try: return json.load(open(cache_path, "r"))
        except: pass

    url_index = "https://publicdata.caida.org/datasets/as-organizations/"
    res = subprocess.check_output(["curl", "-s", url_index]).decode()
    res = re.sub(r"\s\s+", " ", res.replace("\n", " "))
    time2url = {}
    for fname, time in re.findall(r'\<a href="((\d{8}).as-org2info.txt.gz)"\>', res):
        time2url[time] = urljoin(url_index, fname)

    json.dump(time2url, open(cache_path, "w"), indent=2)
    return time2url

def get_most_recent(time):
    time2url = get_archive_list()
    times = sorted(time2url.keys())
    idx = np.searchsorted(times, time, "right")

    target_time = times[idx-1]
    target_url = time2url[target_time]

    out = OUTPUT_DIR/target_url.split("/")[-1]
    if out.with_suffix("").exists():
        # print(f"as-organizations for {target_time} exists")
        return target_time, out.with_suffix("")

    subprocess.run(["curl", target_url, "--output", str(out)], check=True)
    subprocess.run(["gzip", "-d", str(out)], check=True)
    print(f"get as-organizations for {target_time}")
    return target_time, out.with_suffix("")

def download_and_unzip(t, url):
    out = OUTPUT_DIR / url.split("/")[-1]
    unzipped = out.with_suffix("")
    if not unzipped.exists():
        print(f"[INFO] Downloading and Unzipping {t}")
        subprocess.run(["curl", "-s", url, "--output", str(out)], check=True)
        subprocess.run(["gzip", "-df", str(out)], check=True)
    return t, unzipped

def load_as_org_file(filepath):
    field1 = "aut|changed|aut_name|org_id|opaque_id|source".split("|")
    field2 = "org_id|changed|name|country|source".split("|")
    as_info = {}
    org_info = {}

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            values = line.strip().split("|")
            if len(values) == len(field1):
                if values[0] in as_info and values[1] < as_info[values[0]]["changed"]:
                    continue
                as_info[values[0]] = dict(zip(field1[1:], values[1:]))
            elif len(values) == len(field2):
                if values[0] in org_info and values[1] < org_info[values[0]]["changed"]:
                    continue
                org_info[values[0]] = dict(zip(field2[1:], values[1:]))
    return as_info, org_info

def build_as_org_snapshot(time: str, window: int = 6):
    time2url = get_archive_list()
    times = sorted(time2url.keys(), key=int)
    idx = np.searchsorted(times, time, side="right")
    selected_times = times[max(0, idx - window):idx][::-1]  # 从最近到旧的
    print(f"Selected times: {selected_times}")
    
    merged_as_info = {}
    merged_org_info = {}

    for t in selected_times:
        print(f"Processing Org file in {t}")
        t, path = download_and_unzip(t, time2url[t])
        as_info, org_info = load_as_org_file(path)

        for asn, info in as_info.items():
            if (asn in merged_as_info) and (info["changed"] < merged_as_info[asn]["changed"]): continue
            merged_as_info[asn] = info
        for org_id, info in org_info.items():
            if (org_id in merged_org_info) and (info["changed"] < merged_org_info[org_id]["changed"]): continue
            merged_org_info[org_id] = info
        
        with open(OUTPUT_DIR/f"as_info_{t}.txt", "w") as f:
            f.write(json.dumps(merged_as_info, indent=2))

        with open(OUTPUT_DIR/f"org_info_{t}.txt", "w") as f:
            f.write(json.dumps(merged_org_info, indent=2))

    return merged_as_info, merged_org_info

from datetime import datetime

def new_build_as_org_snapshot(time: str, window: int = 6):
    """
    构建AS和Org快照，每个季度只选取一个文件
    
    参数:
    - time: 当前时间，格式为 "YYYYMMDD"
    - window: 从当前时间往前选取的年份数，默认为6年
    
    返回:
    - merged_as_info: 合并后的AS信息
    - merged_org_info: 合并后的Org信息
    """
    # 辅助函数：根据月份获取季度
    def get_quarter(month):
        if 1 <= month <= 3:
            return 1
        elif 4 <= month <= 6:
            return 2
        elif 7 <= month <= 9:
            return 3
        elif 10 <= month <= 12:
            return 4
        else:
            raise ValueError("Invalid month")
    
    # 主要逻辑开始
    time2url = get_archive_list()
    all_times = sorted(time2url.keys(), key=int)
    
    # 将time字符串转换为datetime对象
    current_date = datetime.strptime(time, "%Y%m%d")
    
    # 计算需要包含的最小年份
    min_year = current_date.year - window
    
    selected_quarterly_times = {}
    
    # 遍历所有可用时间，按季度筛选
    for t_str in all_times:
        t_date = datetime.strptime(t_str, "%Y%m%d")
        # 筛选在指定年份范围内且不晚于当前日期的文件
        if t_date.year >= min_year and t_date <= current_date:
            year = t_date.year
            quarter = get_quarter(t_date.month)
            
            # 确保每个季度只选择一个文件，优先选择最新的
            if (year, quarter) not in selected_quarterly_times or t_date > datetime.strptime(selected_quarterly_times[(year, quarter)], "%Y%m%d"):
                selected_quarterly_times[(year, quarter)] = t_str
    
    # 将选中的时间点按降序排列（从最近到最旧）
    selected_times = sorted(selected_quarterly_times.values(), key=int, reverse=True)
    
    print(f"Selected times: {selected_times}")
    
    merged_as_info = {}
    merged_org_info = {}
    
    # 处理每个选中的时间点
    for t in selected_times:
        print(f"Processing Org file in {t}")
        # 检查文件是否存在于time2url中，避免获取不存在的文件
        if t in time2url:
            t, path = download_and_unzip(t, time2url[t])
            as_info, org_info = load_as_org_file(path)
            
            # 合并AS信息，保留最新的changed时间戳
            for asn, info in as_info.items():
                if (asn in merged_as_info) and (info["changed"] < merged_as_info[asn]["changed"]): 
                    continue
                merged_as_info[asn] = info
            
            # 合并Org信息，保留最新的changed时间戳
            for org_id, info in org_info.items():
                if (org_id in merged_org_info) and (info["changed"] < merged_org_info[org_id]["changed"]): 
                    continue
                merged_org_info[org_id] = info
            
            # # 保存中间结果到文件
            # with open(OUTPUT_DIR/f"as_info_{t}.txt", "w") as f:
            #     f.write(json.dumps(merged_as_info, indent=2))
                
            # with open(OUTPUT_DIR/f"org_info_{t}.txt", "w") as f:
            #     f.write(json.dumps(merged_org_info, indent=2))
        else:
            print(f"Warning: File for time {t} not found in archive list. Skipping.")
    
    return merged_as_info, merged_org_info



# @click.command()
# @click.option("--time", "-t", type=str, required=True, help="timestamp, e.g., 20200901")
# def main(time):
#     get_most_recent(time)

@click.command()
@click.option("--time", "-t", type=str, required=True, help="timestamp, e.g., 20200901")
def main(time):
    as_info, org_info = build_as_org_snapshot(time, window=10)

    # 查询一个 ASN 的 org 名称
    asn = "17557"
    if asn in as_info:
        org_id = as_info[asn]["org_id"]
        org_name = org_info.get(org_id, {}).get("name", "Unknown")
        print(f"ASN {asn} → Org: {org_name}")
    else:
        print(f"ASN {asn} not found in snapshot.")

if __name__ == "__main__":
    main()

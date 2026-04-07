#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from io import StringIO
from urllib.parse import urljoin
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import subprocess
import re
import json
import click

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

current_ym = datetime.now().strftime("%Y.%m")
for cache_file in CACHE_DIR.glob(f"*{current_ym}*"): # remove incomplete cache files
    cache_file.unlink() 

def get_all_collectors(url_index="http://routeviews.org/"):
    cache_path = CACHE_DIR/f"collectors2url.{url_index.replace('/', '+')}"
    if cache_path.exists():
        # print(f"load cache: {cache_path}")
        try: return json.load(open(cache_path, "r"))
        except: pass


    res = subprocess.check_output(["curl", "-s", url_index]).decode()
    res = re.sub(r"\s\s+", " ", res.replace("\n", " "))
    collectors2url = {}
    for a, b in re.findall(r'\<A HREF="(.+?)"\>.+?\([\w\s]+, from (.+?)\)', res):
        collector_name = b.split(".")[-3]
        print(f"collector: {collector_name}")
        if collector_name in collectors2url:
            idx = 2
            while f"{collector_name}{idx}" in collectors2url:
                idx += 1
            collector_name = f"{collector_name}{idx}"
        collectors2url[collector_name] = urljoin(url_index, a) + "/"

    # print(f"save cache: {cache_path}")
    json.dump(collectors2url, open(cache_path, "w"), indent=2)
    return collectors2url

def get_archive_list(collector, collectors2url, dtime1, dtime2, type="updates"):
    if collector not in collectors2url: return []

    def pull_list(ym):
        target_url = urljoin(collectors2url[collector], f"{ym}/{type.upper()}") + "/"
        cache_path = CACHE_DIR/f"archive_list.{target_url.replace('/', '+')}"
        if cache_path.exists():
            # print(f"load cache: {cache_path}")
            try: return target_url, json.load(open(cache_path, "r"))
            except: pass
        
        try:
            res = subprocess.check_output(
                ["wget", "-q", "-O", "-", target_url],
                timeout=60
            ).decode()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # print(e)
            print(f"Failed to wget {target_url}")
            try:
                res = subprocess.check_output(
                    ["curl", "-s", target_url],
                ).decode()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(e)      
                exit()

    
        archive_list = re.findall(
            r'\<a href="(.+?(\d{4}).??(\d{2}).??(\d{2}).??(\d{4}).*?\.bz2)"\>', res)
        # print(f"save cache: {cache_path}")
        json.dump(archive_list, open(cache_path, "w"), indent=2)
        return target_url, archive_list

    ym1 = dtime1.strftime("%Y.%m")
    ym2 = dtime2.strftime("%Y.%m")
    target_url1, archive_list1 = pull_list(ym1)
    target_url2, archive_list2 = pull_list(ym2)

    if not archive_list1 or not archive_list2:
        print(f"None Data for archive list: {dtime1} {dtime2}")
        return []
        # exit(1)
    
    time_list1 = ["".join(i[1:]) for i in archive_list1]
    time_list2 = ["".join(i[1:]) for i in archive_list2]
    t1 = dtime1.strftime("%Y%m%d%H%M")
    t2 = dtime2.strftime("%Y%m%d%H%M")
    idx1 = np.searchsorted(time_list1, t1, side="left")
    idx2 = np.searchsorted(time_list2, t2, side="right")

    if time_list1 == time_list2:
        data = [urljoin(target_url1, i[0]) for i in archive_list1[idx1:idx2]]
    else:
        data = [urljoin(target_url1, i[0]) for i in archive_list1[idx1:]]

        current_month = datetime(dtime1.year, dtime1.month, 1)
        current_month += relativedelta(months=1)
        upper_bound = datetime(dtime2.year, dtime2.month, 1)
        while current_month < upper_bound:
            cur_ym = current_month.strftime("%Y.%m")
            cur_target_url, cur_archive_list = pull_list(cur_ym)
            data += [urljoin(cur_target_url, i[0]) for i in cur_archive_list]
            current_month += relativedelta(months=1)
        data += [urljoin(target_url2, i[0]) for i in archive_list2[:idx2]]

    return data

def get_ribs_in_range(collector, collectors2url, start_dt, end_dt):
    if collector not in collectors2url:
        return []

    rib_subdir = "/RIBS"
    target_url = urljoin(collectors2url[collector], f"{start_dt.strftime('%Y.%m')}/{rib_subdir}") + "/"
    cache_path = CACHE_DIR / f"rib_archive_list.{target_url.replace('/', '+')}"

    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                archive_list = json.load(f)
        except:
            archive_list = []
    else:

        res = subprocess.check_output(["curl", "-s", target_url]).decode()
        archive_list = re.findall(r'<a href="(rib\.\d{8}\.\d{4}.*?\.bz2)"', res)
        archive_list = [(fname, ) for fname in archive_list]  
        with open(cache_path, "w") as f:
            json.dump(archive_list, f, indent=2)
        
    if not archive_list or len(archive_list) == 0:
        print(f"No RIB data available for collector {collector} between {start_dt} and {end_dt}")
        return []

    ribs_files = []
    for entry in archive_list:
        fname = entry[0]
        m = re.search(r"(\d{8})\.(\d{4})", fname)
        if m:
            file_time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
            if start_dt <= file_time <= end_dt:
                ribs_files.append(urljoin(target_url, fname))

    return ribs_files

def get_most_recent_rib(collector, collectors2url, dtime):
    if collector not in collectors2url: return []

    def pull_list():
        target_url = urljoin(collectors2url[collector], f"{ym}{subdir}") + "/"
        cache_path = CACHE_DIR/f"archive_list.{target_url.replace('/', '+')}"
        if cache_path.exists():
            # print(f"load cache: {cache_path}")
            try: return target_url, json.load(open(cache_path, "r"))
            except: pass
        res = subprocess.check_output(["curl", "-s", target_url]).decode()
        archive_list = re.findall(
            r'\<a href="(.+?(\d{4}).??(\d{2}).??(\d{2}).??(\d{4}).*?\.bz2)"\>', res)
        # print(f"save cache: {cache_path}")
        json.dump(archive_list, open(cache_path, "w"), indent=2)
        return target_url, archive_list

    ym = dtime.strftime("%Y.%m")
    subdir = "/RIBS"
    target_url, archive_list = pull_list()

    if not archive_list:
        subdir = ""
        target_url, archive_list = pull_list()
    if not archive_list: return []
    
    time_list = ["".join(i[1:]) for i in archive_list]
    t = dtime.strftime("%Y%m%d%H%M")
    idx = np.searchsorted(time_list, t)

    if idx == 0:
        data1 = urljoin(target_url, archive_list[0][0])
        dtime = dtime-relativedelta(months=1)
        ym = dtime.strftime("%Y.%m")
        target_url, archive_list = pull_list()
        if not archive_list: return []
        data0 = urljoin(target_url, archive_list[-1][0])
        stime = datetime.strptime("".join(archive_list[-1][1:]), "%Y%m%d%H%M")
        return data0, data1, stime

    if idx == len(time_list):
        data0 = urljoin(target_url, archive_list[-1][0])
        stime = datetime.strptime("".join(archive_list[-1][1:]), "%Y%m%d%H%M")
        dtime = dtime+relativedelta(months=1)
        ym = dtime.strftime("%Y.%m")
        target_url, archive_list = pull_list()
        if not archive_list: return []
        data1 = urljoin(target_url, archive_list[0][0])
        return data0, data1, stime

    data0 = urljoin(target_url, archive_list[idx-1][0])
    data1 = urljoin(target_url, archive_list[idx][0])
    stime = datetime.strptime("".join(archive_list[idx-1][1:]), "%Y%m%d%H%M")
    return data0, data1, stime

import time
def download_data(url, collector, data_type='updates', output_dir=None):
    fname = url.split("/")[-1].strip()
    if output_dir is None:
        outpath = SCRIPT_DIR / data_type / collector / fname
    else:
        outpath = Path(output_dir) / collector / fname
    fpath = outpath.with_suffix("") 

    if fpath.exists():
        print(f"{data_type} for {collector} {outpath.stem} already existed")
        return fpath

    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[{data_type}] Trying Wget to download {url}")
        subprocess.run(
            ["wget", "-q", "-O", str(outpath), url],
            check=True
        )
    except subprocess.CalledProcessError as e:
        # 尝试 wget 下载（1 次）
        print(f"[{data_type}] Wget Failed, trying curl to download {url}")
        try:
            subprocess.run(
                ["curl", "--fail", "-s", url, "--output", str(outpath)],
                check=True
            )
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(f"Both curl and wget failed to download {url}") from e2

    try:
        subprocess.run(["bzip2", "-d", str(outpath)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"bzip2 decompression failed for {outpath}") from e

    return fpath

def load_updates_to_df(fpath, bgpd=SCRIPT_DIR/"bgpd"):
    try:
        res = subprocess.check_output([str(bgpd), "-q", "-m", "-u", str(fpath)]).decode()
    except:
        print(f"failed to load updates: {fpath}")
        return pd.DataFrame()
    fmt = "type|timestamp|A/W|peer-ip|peer-asn|prefix|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"
    cols = fmt.split("|")
    df = pd.read_csv(StringIO(res), sep="|", names=cols, usecols=cols[:-2], dtype=str, keep_default_na=False)

    df = df.drop_duplicates(subset=["timestamp", "peer-asn", "prefix", "as-path"])
    return df

def load_ribs_to_df(fpath):
    bgpd = SCRIPT_DIR / 'bgpd'
    try:
        res = subprocess.check_output([str(bgpd), "-q", "-m", "-u", str(fpath)]).decode()
    except:
        print(f"failed to load ribs: {fpath}")
        return pd.DataFrame()
    first_line = res.splitlines()[0]
    if first_line.startswith("TABLE_DUMP2_AP"):
        fmt = "type|timestamp|A/W|peer-ip|peer-asn|prefix|metric|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1"
    else:
        fmt = "type|timestamp|A/W|peer-ip|peer-asn|prefix|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"
    cols = fmt.split("|")
    df = pd.read_csv(StringIO(res), sep="|", names=cols, usecols=cols[:-2], dtype=str, keep_default_na=False)
    df = df.drop_duplicates(subset=["timestamp", "peer-asn", "prefix", "as-path"])

    return df

def load_raw_to_df(fpath, bgpd = SCRIPT_DIR / 'bgpd'):
    # Define both formats
    fmt_ap = "type|timestamp|A/W|peer-ip|peer-asn|prefix|metric|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"
    fmt_default = "type|timestamp|A/W|peer-ip|peer-asn|prefix|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"

    expected_cols_ap = len(fmt_ap.split("|"))  # 17
    expected_cols_default = len(fmt_default.split("|"))  # 16

    try:
        res = subprocess.check_output([str(bgpd), "-q", "-m", "-u", str(fpath)]).decode()
    except Exception as e:
        print(f"Failed to load ribs: {fpath}, Error: {e}")
        return pd.DataFrame()

    lines = res.strip().split("\n")
    valid_lines = []
    fallback_lines = []
    invalid_lines = []

    use_fmt = fmt_ap if lines[0].startswith("TABLE_DUMP2_AP") else fmt_default
    expected_col_num = expected_cols_ap if lines[0].startswith("TABLE_DUMP2_AP") else expected_cols_default

    print(f"Using format: {'TABLE_DUMP2_AP' if use_fmt == fmt_ap else 'default'}")
    print(f"Expected columns: {expected_col_num}")

    for idx, line in enumerate(lines):
        col_count = line.count("|") + 1
        if "|W|" in line:
            continue
        if use_fmt:
            if col_count == expected_col_num:
                valid_lines.append(line)
            elif col_count == expected_cols_default:
                # Fallback line: treat as default format (missing one column, e.g., missing 'metric')
                fallback_lines.append(line)
            else:
                invalid_lines.append((idx, col_count, line))
        else:
            valid_lines.append(line)

    # Log malformed lines
    if use_fmt and invalid_lines:
        print(f"[Warning] Found {len(invalid_lines)} malformed lines (not used):")
        for idx, count, line in invalid_lines[:5]:  # only print first 5 for brevity
            print(f"  Line {idx}: expected {expected_col_num}, found {count} -> {line}")

    # Parse valid and fallback lines separately
    frames = []
    if len(valid_lines) == 0 and len(fallback_lines) == 0:
        print(f"[Error] No valid lines found in {fpath}.")
        return pd.DataFrame()

    if valid_lines:
        cols = use_fmt.split("|")
        df_valid = pd.read_csv(
            StringIO("\n".join(valid_lines)),
            sep="|", names=fmt_ap.split("|") if use_fmt == fmt_ap else fmt_default.split("|"),
            usecols=cols[:-2],
            dtype=str, keep_default_na=False
        )
        frames.append(df_valid)

    if use_fmt and fallback_lines:
        print(f"[Info] Found {len(fallback_lines)} fallback lines with one column missing, using default format.")
        cols = fmt_default.split("|")
        df_fallback = pd.read_csv(
            StringIO("\n".join(fallback_lines)),
            sep="|", names=fmt_default.split("|"),
            usecols=cols[:-2],
            dtype=str, keep_default_na=False
        )
        frames.append(df_fallback)

    # Combine both DataFrames if needed
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp", "peer-asn", "prefix", "as-path"])
    else:
        df = pd.DataFrame()

    return df
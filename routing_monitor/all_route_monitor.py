#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import click
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ipaddress import ip_network
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.routeviews.fetch_archive import (
    get_all_collectors,
    get_archive_list,
    download_data,
    get_ribs_in_range,
    load_raw_to_df,
)

from llmmonitor import Monitor

SCRIPT_DIR = Path(__file__).resolve().parent
HISTORY_DIR = SCRIPT_DIR / "__route_cache__"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def detect_archive_files(
    collectors, all_collector_files, route_change_dir, snapshot_dir, monitor, data_type
):
    anchor_collector = "wide" if "wide" in collectors else collectors[0]

    anchor_files = [f for f in all_collector_files if f.parts[-2] == anchor_collector]
    print(f"Found {len(anchor_files)} {anchor_collector} collector files")

    segment_times = []
    for f in anchor_files:
        parts = f.name.split(".")
        try:
            if len(parts) >= 3:
                date, time_str = parts[1], parts[2]
                dt = datetime.strptime(date + time_str, "%Y%m%d%H%M")
                segment_times.append(dt)
        except Exception as e:
            print(f"Failed to parse time from {f.name}: {e}")
            continue
    segment_times = sorted(set(segment_times))

    if not segment_times:
        print("No wide collector time segments found.")
        return

    segment_bounds = []
    for i in range(len(segment_times)):
        start = segment_times[i]
        end = segment_times[i + 1] if i + 1 < len(segment_times) else start + timedelta(minutes=15)
        segment_bounds.append((start, end))

    segment_data = {s[0]: [] for s in segment_bounds}
    for f in tqdm(all_collector_files, desc="Loading data files"):
        try:
            # df = load_updates_to_df(f) if data_type == "updates" else load_ribs_to_df(f)
            df = load_raw_to_df(f)
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            continue
        
        if df is None or df.empty:
            continue
        df = df.drop_duplicates()
        # df["_ts"] = pd.to_datetime(df["timestamp"], unit="s")
        df["_ts"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s")

        for start, end in segment_bounds:
            mask = (df["_ts"] >= start) & (df["_ts"] < end)
            df_seg = df[mask].drop(columns=["_ts"])
            if not df_seg.empty:
                segment_data[start].append(df_seg)

    total_changes = 0
    for i, (start, end) in enumerate(tqdm(segment_bounds, desc="Detecting route changes")):
        date_str = start.strftime("%Y%m%d")
        time_str = start.strftime("%H%M")
        key_str = f"{date_str}.{time_str}"

        out_file = route_change_dir / f"{key_str}.csv"
        if out_file.exists():
            print(f"Skipping existing: {out_file.name}")
            continue

        dfs = segment_data[start]
        if not dfs:
            continue

        df_all = pd.concat(dfs).drop_duplicates().sort_values(by="timestamp")
        monitor.consume(df_all, detect=True)

        route_change_df = pd.DataFrame.from_records(monitor.route_changes)
        monitor.route_changes = []  
        route_change_df.to_csv(out_file, index=False)
        total_changes += len(route_change_df)

        is_last = False
        if i == len(segment_bounds) - 1:
            is_last = True
        else:
            next_date = segment_bounds[i + 1][0].date()
            if next_date != start.date():
                is_last = True
        if is_last:
            pickle.dump(monitor, open(snapshot_dir / f"{date_str}.end-of-the-day", "wb"))

    print(f"Total route: {monitor.route_count}")
    print(f"Total route changes detected: {total_changes}")
    print(f"Total time segments processed: {len(segment_bounds)}")


@click.command()
@click.option("--data-type", "-t", default="updates", type=click.Choice(["updates", "ribs"]))
@click.option("--year", "-y", type=int, required=True)
@click.option("--month", "-m", type=int, required=True)
@click.option("--day", "-d", type=int, required=True)
@click.option("--hour", "-H", type=int, default=12)
@click.option("--minute", "-M", type=int, default=0)
@click.option("--time-range", "-r", type=int, default=12)
@click.option("--num-workers", type=int, default=2)
def main(data_type, year, month, day, hour, minute, time_range, num_workers):
    target_time = datetime(year, month, day, hour, minute)
    file_time = datetime(year, month, day)
    click.echo(f"Target time: {target_time.strftime('%Y-%m-%d %H:%M')}")

    start_time = target_time - timedelta(hours=time_range)
    end_time = target_time + timedelta(hours=time_range)
    click.echo(f"Window: {start_time} to {end_time}")

    result_dir = SCRIPT_DIR / "detection_result" / "all_collectors"

    route_change_dir = result_dir / f"route_change_{target_time.strftime('%Y%m%d%H%M')}"
    snapshot_dir = result_dir / f"snapshot_{target_time.strftime('%Y%m%d%H%M')}"

    route_change_dir.mkdir(exist_ok=True, parents=True)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    detected_days = set()  
    for dt in pd.date_range(start_time, end_time):
        detected_days.add(dt.strftime("%Y%m%d"))
    detected_days = sorted(detected_days)
    for f in snapshot_dir.glob("*.end-of-the-day"):
        if f.name.endswith(".end-of-the-day"):
            date_str = f.name.split(".")[0]
            if date_str in detected_days:
                detected_days.remove(date_str)
                print(f"Detected {date_str} already exists. Skipping.")
    if len(detected_days) == 0:
        click.echo(f"All snapshots already exist. Skipping detection.")
        return

    collectors2url = get_all_collectors()
    collectors = list(collectors2url.keys())

    all_archive_urls = []
    for collector in collectors[:]:
        print(f"Processing collector: {collector}")
        urls = get_archive_list(collector, collectors2url, start_time, end_time, data_type)
        if len(urls) > 0:
            all_archive_urls.extend([(collector, url) for url in urls])
        else:
            print(f'Removing collector {collector} from list.')
            collectors.remove(collector)
    
    click.echo(f"Total {len(all_archive_urls)} {data_type} files to download.")

    hist_start = start_time - timedelta(hours=2)
    hist_end = start_time
    baseline_dfs = []

    cols = ["timestamp", "prefix", "peer-asn", "as-path"]
    for collector in collectors[:]:
        print(f"Processing collector: {collector} for baseline RIB")
        rib_urls = get_ribs_in_range(collector, collectors2url, hist_start, hist_end)
        if not rib_urls or len(rib_urls) == 0:
            continue
        rib_urls = sorted(rib_urls)
        best_url = rib_urls[-1]
        fpath = download_data(best_url, collector, "ribs", str(HISTORY_DIR))
        df = load_raw_to_df(fpath)
        if df is None or df.empty or not all(col in df.columns for col in cols):
            print(f"Failed to load RIB data from {collector}. Removing collector from list.")
            collectors.remove(collector)
            continue
        baseline_dfs.append(df)
    
    print(f"Total {len(baseline_dfs)} RIB files for baseline.")
    full_baseline_df = pd.concat(baseline_dfs).drop_duplicates(subset=["timestamp", "peer-asn", "prefix", "as-path"]).sort_values(by="timestamp")

    if full_baseline_df is None or full_baseline_df.empty:
        click.echo("No valid RIB baseline data found. Aborting.")
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(
            lambda pair: download_data(pair[1], pair[0], data_type, str(HISTORY_DIR)),
            all_archive_urls,
        )
    
    all_files = [
        download_data(pair[1], pair[0], data_type, str(HISTORY_DIR)) for pair in all_archive_urls
    ]

    print(f"Construct baseline with {len(full_baseline_df)} records.")
    monitor = Monitor()
    monitor.load_baseline_from_rib(full_baseline_df)
    click.echo("Stable baseline constructed using all collector RIBs.")

    detect_archive_files(
        collectors,all_files, route_change_dir, snapshot_dir, monitor, data_type
    )
    click.echo("Detection completed.")


if __name__ == "__main__":
    main()

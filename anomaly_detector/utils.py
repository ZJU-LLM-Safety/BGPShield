import pandas as pd
import numpy as np
import json
from pathlib import Path
from functools import lru_cache
from scipy.special import softmax
from itertools import chain
from ipaddress import IPv4Network
import pickle
import math

import torch
import os
def find_free_gpus():
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i)  
        memory_reserved = torch.cuda.memory_reserved(i)   
        if memory_allocated == 0 and memory_reserved == 0:
            free_gpus.append(i)
    return free_gpus

free_gpus = find_free_gpus()
max_gpu = 3
available_gpu = min(max_gpu, len(free_gpus))
if available_gpu > 0:
    print(f"Using GPU: {free_gpus[:available_gpu]}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in free_gpus[:available_gpu])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def read_csv_empty(*args, **kwargs):
    try: return pd.read_csv(*args, **kwargs)
    except pd.errors.EmptyDataError: return pd.DataFrame()

import scipy.stats as stats

def approx_knee_point(x):
    x, y = np.unique(x, return_counts=True) 
    _x = (x-x.min())/(x.max()-x.min())  
    _y = y.cumsum()/y.sum()            
    idx = np.argmax(np.abs(_y-_x))      
    return x[idx], _y[idx]  

def approx_knee_point_continuous(x):
    x = np.array(x)
    
    kde = stats.gaussian_kde(x, bw_method='scott')  
    x_smooth = np.linspace(x.min(), x.max(), 500)  
    y_smooth = kde(x_smooth)  

    cdf_smooth = np.cumsum(y_smooth) / np.sum(y_smooth)

    x_norm = (x_smooth - x_smooth.min()) / (x_smooth.max() - x_smooth.min())

    idx_knee = np.argmax(np.abs(cdf_smooth - x_norm))

    x_knee = x_smooth[idx_knee]
    cdf_knee = cdf_smooth[idx_knee]

    return x_knee, cdf_knee

def load_embs_distance_optim(emb_dir_1, bge=False, return_emb=False):
    emb_dir_1 = Path(emb_dir_1)
    node1_emb_path = emb_dir_1 / "ases_knowledge_info_base_embd.emb"
    node1_emb = pickle.load(open(node1_emb_path, "rb"))
    node1_emb = {
        k: np.array(v, dtype=np.float32)
        for k, v in node1_emb.items()
    }

    @lru_cache(maxsize=300000)
    def _max_emb_distance(a, b): 
        a_list = [as_name.strip() for as_name in a.strip("{}").split(",") if as_name.strip()]
        b_list = [as_name.strip() for as_name in b.strip("{}").split(",") if as_name.strip()]
        max_distance, max_a, max_b = -np.inf, None, None
        found_valid = False
        for as_a in a_list:
            as_a = as_a.strip()
            for as_b in b_list:
                as_b = as_b.strip()
                if as_a == as_b:
                    distance = 0.
                elif as_a in node1_emb and as_b in node1_emb:
                    xi, xj = node1_emb[as_a], node1_emb[as_b]
                    if bge:
                        distance = -np.log(xi @ xj.T + 1e-6)
                    else:
                        distance = np.linalg.norm(xi - xj)
                else:
                    distance = np.inf
                found_valid = True
                if distance > max_distance:
                    max_distance = distance
                    max_a, max_b = as_a, as_b
        if not found_valid:
            return np.inf, None, None
        return max_distance, max_a, max_b

    def emb_distance(a, b):
        return _max_emb_distance(str(a), str(b))

    @lru_cache(maxsize=300000)
    def _min_dtw_distance(s, t):
        s_orig, t_orig = list(s), list(t)
        s_clean = [v for i, v in enumerate(s_orig) if i == 0 or v != s_orig[i-1]]
        t_clean = [v for i, v in enumerate(t_orig) if i == 0 or v != t_orig[i-1]]
        n, m = len(s_clean), len(t_clean)
        DTW = [[(np.inf, [], []) for _ in range(m+1)] for _ in range(n+1)]
        DTW[0][0] = (0.0, [], [])
        for i in range(n):
            for j in range(m):
                cost, rep_s, rep_t = emb_distance(s_clean[i], t_clean[j])
                candidates = [DTW[i][j], DTW[i][j+1], DTW[i+1][j]]
                best = min(candidates, key=lambda x: x[0])
                new_cost = cost + best[0]
                DTW[i+1][j+1] = (
                    new_cost, 
                    best[1] + [(i, rep_s)], 
                    best[2] + [(j, rep_t)]
                )
        final_cost, aligned_s, aligned_t = DTW[n][m]

        def build_orig_to_clean_map(orig):
            orig_to_clean = [None] * len(orig)
            clean_idx = -1
            last = None
            for i, token in enumerate(orig):
                if i == 0 or token != last:
                    clean_idx += 1
                orig_to_clean[i] = clean_idx
                last = token
            return orig_to_clean

        original_to_clean_s = build_orig_to_clean_map(s_orig)
        original_to_clean_t = build_orig_to_clean_map(t_orig)

        for item in aligned_s:
            s_clean[item[0]] = item[1]
        for item in aligned_t:
            t_clean[item[0]] = item[1]

        aligned_s_str, aligned_t_str = s_orig.copy(), t_orig.copy()
        for index, pos in enumerate(original_to_clean_s):
            aligned_s_str[index] = s_clean[pos]
        for index, pos in enumerate(original_to_clean_t):
            aligned_t_str[index] = t_clean[pos]

        for i in range(len(aligned_s_str)):
            if aligned_s_str[i] not in s_orig[i]:
                print(f"Origin Not Match: \n\taligned_s: {aligned_s} \n\trestored_s: {aligned_s_str} \n\toriginal_s: {s_orig}")

        for i in range(len(aligned_t_str)):
            if aligned_t_str[i] not in t_orig[i]:
                print(f"Origin Not Match: \n\taligned_t: {aligned_t} \n\trestored_t: {aligned_t_str} \n\toriginal_t: {t_orig}")

        aligned_s_str = " ".join(aligned_s_str)
        aligned_t_str = " ".join(aligned_t_str)
        return final_cost, aligned_s_str, aligned_t_str, len(aligned_s)

    def dtw_distance(s, t):
        return _min_dtw_distance(tuple(s), tuple(t))

    @lru_cache(maxsize=100000)
    def _path_emb_length(s):
        d = np.array([emb_distance(a,b)[0] for a,b in zip(s[:-1], s[1:])])
        d = d[(d > 0) & (d < np.inf)]
        return np.nan if d.size == 0 else d.sum()

    def path_emb_length(s):
        return _path_emb_length(tuple(s))

    if return_emb:
        return emb_distance, dtw_distance, path_emb_length, node1_emb
    return emb_distance, dtw_distance, path_emb_length


def root_cause_localize_2set(df, th=0.95):
    set1_asn_cnt, set2_asn_cnt = {}, {}
    for i,j in df[["path1", "path2"]].values:
        set_i = set(i.split(" "))
        set_j = set(j.split(" "))
        set_ij = set_i - set_j
        set_ji = set_j - set_i
        for asn in set_ij:
            if asn not in set1_asn_cnt: set1_asn_cnt[asn] = 1
            else: set1_asn_cnt[asn] += 1
        for asn in set_ji:
            if asn not in set2_asn_cnt: set2_asn_cnt[asn] = 1
            else: set2_asn_cnt[asn] += 1

    set1, cnt1 = list(set1_asn_cnt.keys()), list(set1_asn_cnt.values())
    idx1 = np.argsort(cnt1)[::-1]
    set1 = np.array(set1)[idx1]
    cnt1 = np.array(cnt1)[idx1]

    set2, cnt2 = list(set2_asn_cnt.keys()), list(set2_asn_cnt.values())
    idx2 = np.argsort(cnt2)[::-1]
    set2 = np.array(set2)[idx2]
    cnt2 = np.array(cnt2)[idx2]
   
    rc_1, rc_2 = [], []
    for a,b in zip(set1, cnt1):
        if b/df.shape[0] > th: rc_1.append(a)
    for a,b in zip(set2, cnt2):
        if b/df.shape[0] > th: rc_2.append(a)

    return sorted(rc_1), sorted(rc_2)

def root_cause_localize_1set(df, th=0.95):
    set_asn_cnt = {}
    for i,j in df[["path1", "path2"]].values:
        set_i = set(i.split(" "))
        set_j = set(j.split(" "))
        set_xor = set_i^set_j
        for asn in set_xor:
            if asn not in set_asn_cnt: set_asn_cnt[asn] = 1
            else: set_asn_cnt[asn] += 1

    set_asn, cnt = list(set_asn_cnt.keys()), list(set_asn_cnt.values())
    idx = np.argsort(cnt)[::-1]
    set_asn = np.array(set_asn)[idx]
    cnt = np.array(cnt)[idx]

    rc = []
    for a,b in zip(set_asn, cnt):
        if b/df.shape[0] > th: rc.append(a)

    return sorted(rc)

def link_root_cause(culprit_to_df):
    rcs = list(culprit_to_df.keys())
    dfs = list(culprit_to_df.values())

    def rc_to_set(rc):
        culprit_type, culprit_tuple = rc
        assert culprit_type in ["Prefix", "AS"]
        if culprit_type == "AS":
            culprit_set = set(chain(*culprit_tuple))
        else: # must be "Prefix"
            culprit_set = {IPv4Network(p) for p in culprit_tuple}
        return culprit_type, culprit_set

    def rc_set_related(rc1, rc2):
        t1, set1 = rc1
        t2, set2 = rc2
        if t1 != t2:
            return False
        if t1 == "AS":
            return set1&set2
        else: # t1 and t2 must be "Prefix"
            for i in set1:
                for j in set2:
                    if i.overlaps(j): # check if they overlap
                        return True
                    if i.prefixlen == j.prefixlen: # check if they're two consecutive prefixes
                        return abs((int(i[0])>>(32-i.prefixlen))
                                -(int(j[0])>>(32-j.prefixlen))) <= 1
            return False

    pool = list(map(rc_to_set, rcs))
    group_id = [-1]*len(culprit_to_df)
    id_group = dict()
    next_id = 0
    for i in range(len(culprit_to_df)):
        if group_id[i] == -1: 
            group_id[i] = next_id
            next_id += 1
            id_group[group_id[i]] = [i]
        for j in range(i+1, len(culprit_to_df)):
            if group_id[j] == group_id[i]: continue
            if rc_set_related(pool[i], pool[j]):
                if group_id[j] == -1:
                    group_id[j] = group_id[i]
                    id_group[group_id[i]].append(j)
                else:
                    to_be_merged = id_group.pop(group_id[j])
                    id_group[group_id[i]] += to_be_merged
                    for k in to_be_merged: group_id[k] = group_id[i]
    group_id_set = set(group_id)
    group_id_remapping = dict(zip(group_id_set, range(len(group_id_set))))
    for idx, df in enumerate(dfs):
        df["group_id"] = group_id_remapping[group_id[idx]]
    return id_group, pd.concat(dfs, ignore_index=True)

def event_aggregate(events):
    culprit2eventkey = {}
    eventkey2culprit = {}

    for k,v in events.items():
        rc_1, rc_2 = root_cause_localize_2set(v)
        rc_3 = root_cause_localize_1set(v)
        if rc_1 or rc_2:
            culprit = "AS", (tuple(rc_1), tuple(rc_2))
        elif rc_3:
            culprit = "AS", (tuple(rc_3),)
        else:
            culprit = "Prefix", k
        culprit2eventkey.setdefault(culprit, set()).add(k)
        eventkey2culprit[k] = culprit

    culprit_to_df = {k: pd.concat([events[i] for i in v])
                                        for k, v in culprit2eventkey.items()}
    for k, v in culprit_to_df.items():
        _, culprit_tuple = k
        v["culprit"] = json.dumps(culprit_tuple)
    rc_groups, df = link_root_cause(culprit_to_df)

    return rc_groups, df
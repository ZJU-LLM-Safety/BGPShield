#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import time
import torch
import click
import pickle
import random
import networkx as nx
from pathlib import Path
from tqdm import trange, tqdm
from asrank_download import AsnQuery
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.caida_as_org.fetch_data import build_as_org_snapshot 
from data.caida_as_rel.fetch_data import get as prepare_edge_file

primary_device = "cuda:0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 构建 AS 邻接图与数据结构z
class ASGraphBuilder:
    """
    读取 CAIDA as-rel 文件，构建以下数据结构：
      - asn_list: 所有 AS 编号列表（字符串）
      - asn2idx: AS编号到索引的映射
      - upstreams: 每个 AS 的上游（提供商）集合（存储索引）
      - downstreams: 每个 AS 的下游（客户）集合（存储索引）
      - peers: 每个 AS 的对等集合（存储索引）
      - G: 使用 networkx 构建的无向图（用于计算全局指标）
    """
    def __init__(self, merged_as_info, merged_org_info, as_info_path='./'):
        self.asn_list = []  # list of ASNs (自治系统的编号)
        self.asn2idx = {}   # mapping from ASN to index
        self.asn2org = {}   # mapping from ASN to org
        self.asn_info_list = {}  # list of AS info
        self.upstreams = []   # AS 节点的上游节点
        self.downstreams = []   # AS 节点的下游节点
        self.peers = []
        self.G = nx.Graph()
        self.as_info_path = as_info_path    # path to save as infoself.upstreams = [] # AS 节点的上游节点
        self.get_org_id, self.get_org_name = self.build_org_helpers(merged_as_info, merged_org_info)

        if self.as_info_path is not None: 
            os.makedirs(self.as_info_path, exist_ok=True) 


    def _get_index(self, asn):
        if asn not in self.asn2idx:
            idx = len(self.asn_list)
            self.asn2idx[asn] = idx
            self.asn_list.append(asn)
            self.upstreams.append(set())
            self.downstreams.append(set())
            self.peers.append(set())
            self.G.add_node(idx)
        return self.asn2idx[asn]

    def read_caida_as_rel(self, as_rel_file, noise=0, noiseType='0', seed=42):
        random.seed(seed)
        rel_list = []   # [(as1, as2, rel)]
        existing_rel_set = set()

        # 读取所有合法关系（用于原始图和注入前的数据备份）
        rel_list = []  # [(as1, as2, rel)]
        existing_rel_set = set()  # 用于快速查询是否存在关系

        for line in open(as_rel_file, "r"):
            if line[0] == "#":
                continue
            as1, as2, rel = line.strip().split("|")[:3]
            rel_list.append((as1, as2, rel))
            existing_rel_set.add((as1, as2))
            existing_rel_set.add((as2, as1))  # 无向查重时用

        # 噪声注入
        if noise > 0:
            num_to_modify = int((noise/100) * len(rel_list))
            print(f"[Noise Injection] Type: {noiseType}, Ratio: {noise/100}, Count: {num_to_modify}")

            if noiseType == '0':  # Flip relationship
                indices = random.sample(range(len(rel_list)), num_to_modify)
                for idx in indices:
                    as1, as2, rel = rel_list[idx]
                    if rel == "0":
                        rel_list[idx] = (as1, as2, "-1")
                    elif rel == "-1":
                        rel_list[idx] = (as1, as2, "0")
                    # 若其他值，则跳过

            elif noiseType == '1':  # Add non-existent relationship
                added = 0
                all_asns = list({asn for as1, as2, _ in rel_list for asn in (as1, as2)})
                while added < num_to_modify:
                    as1, as2 = random.sample(all_asns, 2)
                    if (as1, as2) in existing_rel_set or (as2, as1) in existing_rel_set:
                        continue
                    rel = random.choice(["0", "-1"])
                    rel_list.append((as1, as2, rel))
                    existing_rel_set.add((as1, as2))
                    existing_rel_set.add((as2, as1))
                    added += 1

            elif noiseType == '2':  # Delete relationship
                indices = random.sample(range(len(rel_list)), num_to_modify)
                for idx in sorted(indices, reverse=True):
                    as1, as2, _ = rel_list[idx]
                    existing_rel_set.discard((as1, as2))
                    existing_rel_set.discard((as2, as1))
                    rel_list.pop(idx)

            else:
                raise ValueError(f"Unsupported noiseType: {noiseType}")

        # 重新构建图结构
        for as1, as2, rel in rel_list:
            i1 = self._get_index(as1)
            i2 = self._get_index(as2)
            if i1 == i2:
                continue
            if rel == "0":
                self.peers[i1].add(i2)
                self.peers[i2].add(i1)
                self.G.add_edge(i1, i2)
            elif rel == "-1":
                self.downstreams[i1].add(i2)
                self.upstreams[i2].add(i1)
                self.G.add_edge(i1, i2)
            else:
                raise RuntimeError(f"unexpected relationship: {rel}")

        print(f"Total AS nodes: {len(self.asn_list)}")
        print(f"Total edges in G: {self.G.number_of_edges()}")
    
    def _load_asn_info(self, asn_info_file):
        if os.path.exists(asn_info_file):
            try:
                with open(asn_info_file, "rb") as f:
                    return pickle.load(f)
            except:
                print(f"Failed to load {asn_info_file}")
        return {}

    def _save_asn_info(self, asn_info_dict ,asn_info_file):
        with open(asn_info_file, "wb") as f:
            pickle.dump(asn_info_dict, f)

    def construct_as_rank(self, batch_size=10, file_name="all_asn_info.pkl"):
        pkl_file = os.path.join(self.as_info_path, file_name)
        error_file_path = os.path.join(self.as_info_path, f"errorAS_{os.uname()[1]}.txt")
        asn_info_dict = self._load_asn_info(pkl_file)

        for asn in list(asn_info_dict.keys()):
            if asn_info_dict[asn] == {} or asn_info_dict[asn] is None:
                del asn_info_dict[asn]

        total = len(self.asn_list)
        print(f"Constructing AS knowledge for {total} ASes...")
        to_query = [asn for asn in self.asn_list if asn not in asn_info_dict]
        print(f"Total ASes: {total}, Already queried: {total - len(to_query)}, To query: {len(to_query)}")

        if to_query == []:
            print("All ASes have been queried.")
            return True

        error_asns = []
        if os.path.exists(error_file_path):
            return False

        for i in trange(0, len(to_query), batch_size, desc="Querying AS Rank API"):
            batch = to_query[i:i+batch_size]
            batch_results = {}

            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = {executor.submit(AsnQuery, int(asn)): asn for asn in batch}

                for future in as_completed(futures):
                    asn = futures[future]
                    try:
                        result = future.result()
                        if result is None:
                            print(f"Failed to query {asn}")
                            error_asns.append(asn)
                            continue
                        batch_results[asn] = result
                    except Exception as exc:
                        error_asns.append(futures[future])
                        print(f"Error querying {futures[future]}: {exc}")
            
            if len(batch_results) != 0:
                asn_info_dict.update(batch_results)
                self._save_asn_info(asn_info_dict, pkl_file)

        if error_asns:
            with open(error_file_path, "a") as f:
                f.write("\n".join(error_asns) + "\n")
            print(f"{len(error_asns)} ASes failed to query.")
            return False

        print("AS knowledge construction done.")
        return True

    def asAuditAndReQuery(self, batch_size=10 , pkl_file_name="all_asn_info.pkl"):
        error_file_path = os.path.join(self.as_info_path, f"errorAS_{os.uname()[1]}.txt")
        pkl_file = os.path.join(self.as_info_path, pkl_file_name)

        if not os.path.exists(error_file_path):
            print(f"Error file {error_file_path} does not exist.")
            return
        
        with open(error_file_path, "r") as f:
            error_asns = list(set(line.strip() for line in f if line.strip().isdigit()))

        if not error_asns:
            print("No ASes to re-query.")
            return

        print(f"Re-querying {len(error_asns)} ASes...")
        asn_info_dict = self._load_asn_info(pkl_file)
        new_error_asns = []

        for i in tqdm(range(0, len(error_asns), batch_size), desc="Re-querying AS Rank API"):
            batch = error_asns[i:i+batch_size]

            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = {executor.submit(AsnQuery, int(asn)): asn for asn in batch}

                for future in as_completed(futures):
                    asn = futures[future]
                    try:
                        result = future.result()
                        if result is None:
                            print(f"Failed to query {asn}")
                            new_error_asns.append(asn)
                            continue
                        asn_info_dict[asn] = result
                    except Exception as exc:
                        new_error_asns.append(futures[future])
                        print(f"Error querying {futures[future]}: {exc}")

            self._save_asn_info(asn_info_dict, pkl_file)

        if new_error_asns:
            with open(error_file_path, "w") as f:
                f.write("\n".join(new_error_asns) + "\n")
            print(f"{len(new_error_asns)} ASes failed to query.")
            return False

        return True

    def construct_as_org(self):
        print("Querying AS Organization...")
        count = 0
        for asn in self.asn_list:
            org_name, org_country = self.get_org_name(self.get_org_id(asn))
            self.asn2org[asn] = f"{org_name} (Country: {org_country})"
            if org_name.startswith("Org_"):
                count += 1
        print("AS Organization query done.")
        print(f"Org_ count: {count}")

    def asKnoledgeConstruct(self, asn_list=None, num_workers=5, max_retries=5, max_error_as=10):
        """
        Construct the knowledge of ASes from AS Rank API and AS relationship.
        """
        if asn_list is None:
            print("ERROR: batch_asn_list is None")
            exit()
        allowed_workers = min(num_workers, len(asn_list), os.cpu_count() + 4)
        lbound, rbound = self.asn2idx[asn_list[0]], self.asn2idx[asn_list[-1]]
        pklFile = os.path.join(self.as_info_path, f"asn_info_{lbound}_{rbound}.pkl")
        errorFile = os.path.join(self.as_info_path, f"errorAS_{os.uname()[1]}.txt")
        error_as = []   # Record the AS that failed to query
        if os.path.exists(pklFile):
            print(f"File {pklFile} already exists")
            with open(errorFile, 'a') as ef:
                ef.write(f"{lbound}: {error_as}\n")
            return 0
        for retry in range(max_retries):
            error_num = 0   # Record the number of failed queries
            print(f"---------Attempt {retry + 1} for {pklFile}---------")
            if os.path.exists(pklFile): # Delete the file if it exists
                os.remove(pklFile)
            if len(error_as) > 0:
                asn_list = error_as
                error_as = []

            # Request AS Rank API concurrently
            with ThreadPoolExecutor(max_workers=allowed_workers) as executor:
                future_to_asn = {
                    executor.submit(AsnQuery, int(asn)) : asn for asn in asn_list
                }
            for future in as_completed(future_to_asn): 
                asn = future_to_asn[future]
                idx = self.asn2idx[asn]
                try:
                    as_info_tmp = future.result()
                    if as_info_tmp is None:
                        error_num += 1
                        print(f"Warning: Request Rank API failed for {asn}")
                        error_as.append(asn)
                        self.asn_info_list[idx] = {
                            'asn': asn,
                            'asnName': None,
                            'rank': None,
                            'organization': {'orgId': None, 'orgName': None},
                            'cliqueMember': None,
                            'seen': None,
                            'longitude': None,
                            'latitude': None,
                            'cone': {'numberAsns': None, 'numberPrefixes': None, 'numberAddresses': None},
                            'country': {'iso': None, 'name': None},
                            'asnDegree': {'provider': None, 'peer': None, 'customer': None, 'total': None, 'transit': None, 'sibling': None},
                            'announcing': {'numberPrefixes': None, 'numberAddresses': None}
                        }
                    else:
                        print(f"Fetched \t{idx}: \t{asn} info")
                        self.asn_info_list[idx] = as_info_tmp
                    idx = self.asn2idx[asn]
                    self.asn_info_list[idx]['topology information'] = {
                        'p2p': [i for i in self.p2p[idx]],
                        'p2c': [i for i in self.p2c[idx]],
                        'c2p': [i for i in self.c2p[idx]],
                    }
                except Exception as e:
                    error_num += 1
                    print(f"ERROR processing ASN \t{asn}: {e}")
            time.sleep(0.5)  # Sleep for 0.5s before write
            try:
                if error_num < max_error_as:
                    with open(pklFile, 'wb') as f:
                        pickle.dump(self.asn_info_list[lbound:rbound+1], f)
                    f.close()
                    with open(errorFile, 'a') as ef:
                        ef.write(f"{lbound}: {error_as}\n")
                    print(f"Successfully saved {pklFile}")
                    return
            except Exception as e:
                print(f"ERROR for {pklFile}: {e}")
        print(f"ERROR: Failed to process {pklFile} after {max_retries} retries")
        return False

    def old_asAuditAndReQuery(self, num_workers=1, max_retries=10):
        """
        ReQuery the ASes that failed before and the ASes that failed to write to the pkl file.
        (from errorAS*.txt with format "lbound: [error_as_list]")
        """
        error_as = {}   # Record the AS that failed to query
        # Merge all errorAS*.txt files
        errorFilesToMerge = [os.path.join(self.as_info_path, f) for f in os.listdir(self.as_info_path) if f.startswith("errorAS")]
        errorFile = os.path.join(self.as_info_path, f"errorAS.txt")
        for file in errorFilesToMerge:
            print(f"Processing {file}")
            with open(file, 'r') as f:
                for line in f:
                    idx, error_as_list = line.strip().split(": ")
                    error_as_list = ast.literal_eval(error_as_list)
                    rbound = int(idx) + self.batchSize if int(idx) + self.batchSize < len(self.asn_list) else len(self.asn_list)
                    pklFile = os.path.join(self.as_info_path, f"asn_info_{idx}_{rbound-1}.pkl")
                    if not os.path.exists(pklFile):
                        self.asKnoledgeConstruct(asn_list=self.asn_list[int(idx):rbound])
                        continue
                    else:
                        with open(pklFile, 'rb') as pklf:
                            as_info_list_tmp = pickle.load(pklf)
                    
                    # 去除 as_info_list_tmp 中的空字典
                    as_info_list_tmp = [tmp for tmp in as_info_list_tmp if tmp.get('rank') is not None]

                    # Check the data in pkl file is valid (not None)
                    if len(as_info_list_tmp) == rbound-int(idx):
                        print(f"File {os.path.basename(pklFile)} is valid")
                        continue
                    
                    missingNum = rbound-int(idx)-len(as_info_list_tmp)
                    i = 0   # index for as_info_list_tmp
                    j = int(idx)    # index for self.asn_info_list
                    while j < rbound:
                        if i >= len(as_info_list_tmp):
                            # That means the tail ASes are missing
                            for k in range(missingNum):
                                print(f"ERROR: {j} Missing AS \t{self.asn_list[j]} in {pklFile}")
                                error_as_list.append(self.asn_list[j])
                                j += 1
                            break
                        item = as_info_list_tmp[i]
                        curIdx, asIdx = j, self.asn2idx[item.get('asn')]
                        i += 1
                        j += 1
                        if curIdx != asIdx:
                            missingAS = self.asn_list[curIdx]
                            error_as_list.append(missingAS)
                            j += 1
                            missingNum -= 1
                            print(f"ERROR: {curIdx} Missing AS \t{missingAS} in {pklFile}")
                            continue
                        elif item.get('rank') is None:
                            asn_tmp = item.get('asn')
                            error_as_list.append(asn_tmp)
                            print(f"ERROR: {self.asn2idx[asn_tmp]} Invalid ASN \t{asn_tmp} in {pklFile}")
                        self.asn_info_list[asIdx] = item
                    
                    if len(error_as_list) != 0: # if the error_as_list is empty, skip
                        for retry in range(max_retries):
                            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                                future_to_asn = {
                                    executor.submit(AsnQuery, int(asn)) : asn for asn in error_as_list
                                }
                            error_as_tmp = []
                            for future in as_completed(future_to_asn):
                                asn = future_to_asn[future]
                                asn2idx = self.asn2idx[asn]
                                try:
                                    as_info_tmp = future.result()
                                    if as_info_tmp is None:
                                        error_as_tmp.append(asn)
                                        print(f"Warning: Request Rank API failed for {asn}")
                                    else:
                                        print(f"Fetched \t{asn2idx}: \t{asn} info")
                                        self.asn_info_list[asn2idx] = as_info_tmp
                                        self.asn_info_list[asn2idx]['topology information'] = {
                                            'p2p': [i for i in self.p2p[asn2idx]],
                                            'p2c': [i for i in self.p2c[asn2idx]],
                                            'c2p': [i for i in self.c2p[asn2idx]],
                                        }
                                except Exception as e:
                                    error_as_tmp.append(asn)
                                    print(f"ERROR processing ASN \t{asn}: {e}")
                            if len(error_as_tmp) > 0:
                                error_as_list = error_as_tmp
                            else:
                                break
                        if len(error_as_tmp) > 0:
                            error_as += error_as_tmp
                            with open(errorFile, 'a') as ef:
                                ef.write(f"{idx}: {error_as_tmp}\n")
                            print(f"ERROR: Failed to re-query {pklFile}")
                        else:
                            with open(pklFile, 'wb') as pklf:
                                pickle.dump(self.asn_info_list[int(idx):rbound], pklf)
                            pklf.close()
                            print(f"Successfully saved {pklFile}")
            f.close()
            # 重命名为 DONE_errorAS*.txt
            errorFileDone = os.path.join(self.as_info_path, f"DONE_{os.path.basename(file)}")
            os.rename(file, errorFileDone)
        # If dict error_as is not empty, retry the failed ASes
        if len(error_as) > 0:
            print(f"ERROR: Failed to re-query {len(error_as)} ASes")
            with open(errorFile, 'a') as ef:
                for idx, error_as_list in error_as.items():
                    ef.write(f"{idx}: {error_as_list}\n")
            ef.close()
            print(f"Please re-execute the LBEAM.py for requrying the failed ASes")
            exit()
            # self.asReQuery()

    def build_org_helpers(self, as_info, org_info):
        def get_org_id(asn):
            info = as_info.get(asn)
            if info:
                return info["org_id"] if info["org_id"] != "" else info["opaque_id"]
                # return info["opaque_id"] if info["opaque_id"] != "" else info["org_id"]
            # AS 不在 merged_as_info 中时，org_id 用 ASN 自身
            return str(asn)

        def get_org_name(org_id):
            # 先从 merged_org_info 取 name
            org_name = org_info.get(org_id, {}).get("name", "").strip()
            org_country = org_info.get(org_id, {}).get("country", "").strip()
            # 如果没有有效 name，就用 Org_<org_id> 回退
            if not org_name:
                org_name = f"Org_{org_id}"
            if not org_country:
                # 如果没有有效 country，就用 unknown
                org_country = "unknown"
            # name = f"{org_name} (Country: {org_country})"
            return org_name, org_country

        return get_org_id, get_org_name


def compute_global_metrics(G):
    """
    计算全局图指标：
      - 度中心性（degree centrality）
      - 局部聚类系数（clustering coefficient）
      - PageRank
    """
    print("Computing global degree centrality...")
    deg_centrality = nx.degree_centrality(G)
    print("Computing global clustering coefficients...")
    clustering = nx.clustering(G)
    print("Computing global PageRank (this may take a while)...")
    pagerank = nx.pagerank(G)
    return {
        "degree_centrality": deg_centrality,
        "clustering": clustering,
        "pagerank": pagerank,
    }

def build_as_full_description(asn, builder, global_metrics, asrank=False, batch_size=1000):
    """
    为目标AS构造描述信息，包括基本信息和所有邻居描述，区分不同关系类型。
    返回 (as_info, neighbor_batches)
      - as_info: 目标AS的基本信息描述
      - neighbor_batches: 按批次拆分的邻居描述列表
      - batch_size: 每批次邻居数量
    """
    idx = builder.asn2idx.get(asn)
    org_country = builder.asn2org.get(asn, f"ASN_{asn} (Country: unknown)")
    if idx is None:
        return None, None
    if not asrank:
        as_info = f"Overview for AS {asn} From {org_country}:\n"
        as_info += f"1. Global PR (PageRank) is {global_metrics['pagerank'].get(idx, 0):.6f}\n"
        as_info += f"2. Degree in BGP Graph: {builder.G.degree(idx)}\n"
        as_info += (
            f"3. AS{asn}'s Provider Neighbors: {len(builder.upstreams[idx])}, "
            f"Customer Neighbors: {len(builder.downstreams[idx])}, "
            f"Peer Neighbors: {len(builder.peers[idx])}\n\n"
        )
    else:
        info = builder.asn_info_list.get(asn)
        as_info = f"Overview for AS {asn} From {org_country}:\n"
        as_info += f"1. Global PR (PageRank) is {global_metrics['pagerank'].get(idx, 0):.6f}\n"
        as_info += f"2. Degree in BGP Graph: {builder.G.degree(idx)}\n"
        as_info += (
            f"3. AS{asn}'s Provider Neighbors: {len(builder.upstreams[idx])}, "
            f"Customer Neighbors: {len(builder.downstreams[idx])}, "
            f"Peer Neighbors: {len(builder.peers[idx])}\n\n"
        )
        if info is not None:
            # 下面这些字段名根据 AS Rank API 返回结构调整
            # as_name       = info.get("name", f"AS{asn}")
            country       = info.get("country", "unknown")
            org_id        = info.get("org_id", "N/A")
            prefix_count  = info.get("prefixes", 0)
            # asn_created   = info.get("created", "unknown")

                # f"AS Rank Overview for AS {asn} - {as_name}:\n"
            as_info += f"4. Registered Organization ID: {org_id}\n"
            as_info += f"5. Country of Registration: {country}\n"
            as_info += f"6. Number of Announced Prefixes: {prefix_count}\n"
                # f"4. ASN Registration Date: {asn_created}\n\n"


    # 收集邻居信息并区分关系
    neighbor_texts = []
    neighbor_len = len(builder.upstreams[idx]) + len(builder.downstreams[idx]) + len(builder.peers[idx])
    for n in (builder.upstreams[idx] | builder.downstreams[idx] | builder.peers[idx]):
        neighbor_as = builder.asn_list[n]
        neighbor_org = builder.asn2org.get(neighbor_as, f"ASN_{neighbor_as} (Country: unknown)")
        if n in builder.upstreams[idx]:
            rel_label = "Provider (upstream neighbor)"
        elif n in builder.downstreams[idx]:
            rel_label = "Customer (downstream neighbor)"
        elif n in builder.peers[idx]:
            rel_label = "Peer (peer neighbor)"
        else:
            raise ValueError(f"Unknown relationship for AS {asn} and AS {neighbor_as}")
        # neighbor_text = (f"[AS{neighbor_as} ({rel_label}): PR={global_metrics['pagerank'].get(n, 0):.6f}, Degree={builder.G.degree(n)}]")
        neighbor_idx = builder.asn2idx[neighbor_as]
        neighbor_as_cus = len(builder.downstreams[neighbor_idx])
        neighbor_as_pro = len(builder.upstreams[neighbor_idx])
        neighbor_as_peers = len(builder.peers[neighbor_idx])
        neighbor_text = f"AS{asn}'s({rel_label}): AS{neighbor_as} from {neighbor_org} which has {neighbor_as_cus} customer neighbors, {neighbor_as_pro} provider neighbors, {neighbor_as_peers} peers;"
        neighbor_texts.append(neighbor_text)

    
    # 分批次拆分
    neighbor_batches = [neighbor_texts[i:i+batch_size] for i in range(0, len(neighbor_texts), batch_size)]

    return as_info, neighbor_batches, neighbor_len

from FlagEmbedding import FlagModel, BGEM3FlagModel

class ASInfoProcessor:
    def __init__(self, model_path, bge, batch_size=4, precision="float16", device=primary_device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.precision = precision
        if bge:
            self.model = BGEM3FlagModel(
                model_path,  
                # query_instruction_for_retrieval="Generate representations based on a subset of this AS's BGP business neighbors to distinguish routing roles and further apply them for anomaly detection (with neighbors provided in batches).",
                use_fp16=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            # if "Qwen" in model_path:
            #     self.model = AutoModel.from_pretrained(
            #         model_path,
            #         device_map="auto",
            #         low_cpu_mem_usage=True,
            #         quantization_config=quantization_config,
            #         pad_token_id=self.tokenizer.eos_token_id,
            #         trust_remote_code=True
            #     )
            # else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                pad_token_id=self.tokenizer.eos_token_id,
                trust_remote_code=True
            )
            
        self.device = device
        self.batch_size = batch_size
        self.layer = -1

    def batchEmbed(self, sentences):
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
            
            hidden_states = outputs.hidden_states[self.layer]

            batch_emb = hidden_states.mean(dim=1)
            all_embeddings.append(batch_emb)
            del batch, inputs, outputs, hidden_states, batch_emb
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0)

    def CleanBatchEmbed(self, sentences):
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt", return_attention_mask=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
            
            hidden_states = outputs.hidden_states[self.layer]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            masked_hidden = hidden_states * attention_mask
            masked_hidden = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)

            batch_emb = masked_hidden / counts
            all_embeddings.append(batch_emb)
            del batch, inputs, outputs, hidden_states, batch_emb, attention_mask
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0)

    def bertBatchEmbed(self, sentences):
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            with torch.no_grad():
                # batch_emb = self.model.encode(batch)
                batch_emb = self.model.encode(batch,
                                              max_length=8192,
                                              )['dense_vecs']

            all_embeddings.append(batch_emb)
            del batch, batch_emb
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        all_embeddings = [torch.tensor(embed) if not isinstance(embed, torch.Tensor) else embed for embed in all_embeddings]
        return torch.cat(all_embeddings, dim=0)


    def generate_role_explanation(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                pad_token_id=self.tokenizer.eos_token_id, 
                max_new_tokens=128,
                min_new_tokens=64,
                do_sample=True, # TODO
                temperature=0.7,
            )
        explanation = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        del inputs, generated_ids
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return explanation

def generate_batch_summary(prompt, processor):
    """
    通过 LLM 生成当前 prompt 的摘要，并返回摘要的嵌入向量。
    """
    summary_text = processor.generate_role_explanation(prompt)
    summary_emb = processor.batchEmbed([summary_text])[0]
    return summary_text, summary_emb

def iterative_embds_generation(asn, builder, processor, global_metrics, asrank=False, batch_size=50):
    """
    采用迭代式上下文扩充生成目标AS所有邻居的汇总摘要。
    每轮将当前邻居批次加入上下文，并生成新的摘要。
    返回最终摘要文本及其嵌入。
    batch_size: 一个批次的邻居数量
    """
    idx = builder.asn2idx.get(asn)
    if idx is None:
        return None, None

    base_info, neighbor_batches, neighbor_len = build_as_full_description(asn, builder, global_metrics, asrank=asrank, batch_size=batch_size)
    # emb = processor.batchEmbed([base_info])[0].cpu().detach()
    emb = processor.CleanBatchEmbed([base_info])[0].cpu().detach()

    if neighbor_batches is None or len(neighbor_batches)==0:
        return base_info, emb
    
    # print(f"\nBuilding description for AS {asn} DONE!!!")

    # print(f"\nGenerating embeddings for AS {asn} neighbors...")

    final_emb = []
    current_context = base_info
    tmp = neighbor_len
    for batch in neighbor_batches:
        batch_neighbor_len = len(batch)
        tmp = tmp - batch_neighbor_len
        prompt = base_info + \
            f"\n{batch_neighbor_len} Neighbors for AS{asn}({neighbor_len} Neighbors in all):\n" + "\n".join(batch)  +  \
            f"\n, Based on AS{asn}'s global statistics, its direct business relationships, and the size and structure of each neighbor’s network (including their providers, peers, and customers number), infer AS{asn}'s routing role to reflect its routing behavior and policy rationale." + \
            f"Following that, there will be {tmp} additional neighbors’ information for AS{asn}." 
        summary_emb = processor.batchEmbed([prompt])[0]
        final_emb.append(summary_emb.cpu().detach())
        del summary_emb
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # final_summary = current_context
    final_emb = torch.stack(final_emb).mean(dim=0)
    # return final_summary, final_emb
    return final_emb

def bert_iterative_embds_generation(asn, builder, processor, global_metrics, asrank=False, batch_size=50):
    """
    采用迭代式上下文扩充生成目标AS所有邻居的汇总摘要。
    每轮将当前邻居批次加入上下文，并生成新的摘要。
    返回最终摘要文本及其嵌入。
    batch_size: 一个批次的邻居数量
    """
    idx = builder.asn2idx.get(asn)
    if idx is None:
        return None, None

    base_info, neighbor_batches, neighbor_len = build_as_full_description(asn, builder, global_metrics, asrank=asrank, batch_size=batch_size)
    emb = processor.bertBatchEmbed([base_info])[0].cpu().detach()
    if neighbor_batches is None or len(neighbor_batches)==0:
        return base_info, emb
    
    # print(f"\nBuilding description for AS {asn} DONE!!!")

    # print(f"\nGenerating embeddings for AS {asn} neighbors...")

    final_emb = []
    tmp = neighbor_len
    current_context = base_info
    # for batch in tqdm(neighbor_batches, desc="Generating Neighbor Embeddings"):
    for batch in neighbor_batches:
        # 这部分会显存不足，本来是为了衔接分批次传入 LLM 带来的影响（可能影响 LLM 全局解释）
        # prompt = current_context + "\nNeighbors: " + ", ".join(batch)
        # summary_text, summary_emb = generate_batch_summary(prompt, processor)
        # final_emb.append(summary_emb.cpu().detach())
        # current_context = summary_text
        # del summary_text, summary_emb
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        batch_neighbor_len = len(batch)
        tmp = tmp - batch_neighbor_len
        prompt = base_info + \
            f"\n{batch_neighbor_len} Neighbors for AS{asn}({neighbor_len} Neighbors in all):\n" + "\n".join(batch)  +  \
            f"\n, Based on AS{asn}'s global statistics, its direct business relationships, and the size and structure of each neighbor’s network (including their providers, peers, and customers number), infer AS{asn}'s routing role to reflect its routing behavior." + \
            f"Following that, there will be {tmp} additional neighbors’ information for AS{asn}." 
        summary_emb = processor.bertBatchEmbed([prompt])[0]
        final_emb.append(summary_emb.cpu().detach())
        del summary_emb
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    final_emb = torch.stack(final_emb).mean(dim=0)

    return final_emb

@click.command()
@click.option("--year", "-y", type=int, required=True, help="the year of the route changes monitored, e.g., 2024")
@click.option("--month", "-m", type=int, required=True, help="the month of the route changes monitored, e.g., 8")
@click.option("--model", "-M", type=int, default=0, help="id of LLM Model")
@click.option("--asrank", "-r", type=bool, default=False, help="whether to use AS Rank API as global statistics (suggested for latest detection)")
@click.option("--device", "-d", type=str, default="0", help="Comma-separated GPU ids to use, e.g., '0,2,4'")
@click.option("--noise", "-n", type=click.Choice(['0', '5', '10', '15', '20', '25']), default='0', help="Whether to add noise to the input")
@click.option("--noise-type", "-t", type=click.Choice(['0', '1', '2']), default='0', help="The type of noise, 0: FLIP, 1: ADD, 2: DELETE")
def main(year, month, asrank, model, device, noise, noise_type):
    date_str = f"{year:04d}{month:02d}01"
    date = int(date_str)
    serial = "1" if date < 20151201 else "2"

    print(f"Date: {date_str}")
    as_rel_file = prepare_edge_file(serial=serial, time=date_str)
    print(f"Edge file: {as_rel_file}")

    print(f"Prepare Org file...")
    merged_as_info, merged_org_info = build_as_org_snapshot(date_str, window=10)
    print(f"Org files are ready.")

    gpu_list = device.split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    print(f"[INFO] Using GPUs: {gpu_list}")
    
    as_info_path = Path(__file__).resolve().parent/"as_info"/ \
        f"{date_str}"/f"ases_knowledge_info_base" # 保存 AS 的信息,后续重新查询，可以直接在此基础上增添
    # 构建AS图
    builder = ASGraphBuilder(merged_as_info, merged_org_info, as_info_path)
    builder.read_caida_as_rel(as_rel_file, int(noise), noise_type)

    if not asrank:
        # 计算全局指标
        print("Computing global graph metrics...")
        global_metrics = compute_global_metrics(builder.G)
        print("Global metrics computed.")
        builder.construct_as_org()
    else:
        # 使用 AS Rank API 获取全局指标
        print("Using AS Rank API to Construct AS Description...")
        success = builder.construct_as_rank(batch_size=15)
        if not success:
            for i in range(3):
                time.sleep(10)
                print(f"Retry AS Rank API for the {i+1} time...")
                success = builder.asAuditAndReQuery()
                if success:
                    break
        print("AS Rank API Query Finished.")
        exit(0)

    model_list = [
        "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/", 
        "/hub/huggingface/models/Qwen/Qwen3-1.7B-Base/", 
        "/hub/huggingface/models/BAAI/bge-m3/",
        "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
        ]
    
    model_path = model_list[int(model)]

    bge = True if model_path == "/hub/huggingface/models/BAAI/bge-m3/" else False

    model_name = model_path.split("/")[-2]
    print(f"Model: {model_name}")

    # # 初始化LLM处理器（Llama-3-8B）
    # if bge:
    #     model_path = "/hub/huggingface/models/BAAI/bge-m3/"     # 568M  No.20
    #     # model_path = "/hub/huggingface/models/Alibaba-NLP/gte-Qwen2-7B-instruct" # 7B No.3
    #     # model_path = "/hub/huggingface/models/Alibaba-NLP/gte-Qwen2-1.5B-instruct"   # 1.5B No.11
    #     # 
    # else:
    #     # model_path = "/hub/huggingface/models/Qwen/Qwen3-1.7B-Base/" 
    #     # model_path = "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
    #     model_path = "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/"

    # LLM embedding 保存路径
    # embds_dir = Path(__file__).resolve().parent/"llMmodels/newprompt/mean/iterative_as_info"
    embds_dir = Path(__file__).resolve().parent/"llMmodels/moreprompt/mean/iterative_as_info"


    embds_dir = embds_dir / date_str / model_name 
    if asrank:
        embds_dir = embds_dir / "asrank"
    os.makedirs(embds_dir, exist_ok=True)

    os.makedirs(embds_dir, exist_ok=True)
    if noise != '0':
        embds_dir = embds_dir / f"noise_{noise}_{noise_type}"
        os.makedirs(embds_dir, exist_ok=True)

    embds_final = os.path.join(embds_dir, "ases_knowledge_info_base_embd.emb")
    mid_embds = os.path.join(embds_dir, "mid.emb")
    
    if os.path.exists(embds_final):
        print(f"Embeddings already generated for {embds_final}")
        return

    processor = ASInfoProcessor(model_path, bge, batch_size=1, precision="float16", device=primary_device)

    # 对每个AS生成最终embedding
    final_embeddings = {}

    if os.path.exists(mid_embds):
        print("Load mid embeddings...")
        try:
            with open(mid_embds, "rb") as f:
                final_embeddings = pickle.load(f)
            print(f"Loaded {len(final_embeddings)} mid embeddings.")
        except Exception as e:
            print(f"Failed to load mid embeddings: {e}")
            os.remove(mid_embds)
            final_embeddings = {}

    i = 0
    pbar = tqdm(builder.asn_list, desc="AS Embedding Generation")
    for asn in pbar:
        pbar.set_description(f"Processing AS: {asn}")
        # 如果已有中间结果，则跳过
        if asn in final_embeddings:
            continue
        
        # 利用迭代式上下文扩充生成邻域摘要嵌入
        if bge:
            neighbor_emb = bert_iterative_embds_generation(asn, builder, processor, global_metrics, asrank=asrank, batch_size=20)  # 此处 batch_size 为每批次邻居数量， BGE-M3
        else:
            neighbor_emb = iterative_embds_generation(asn, builder, processor, global_metrics, asrank=asrank, batch_size=20)

        if neighbor_emb is None:
            raise RuntimeError(f"Failed to generate embedding for AS {asn}")
        final_embeddings[asn] = neighbor_emb
        i += 1
        # print(f"AS {asn} embedding generated.")
        # 保存中间结果
        if i % 200 == 0:
            print(f"Saving mid embeddings till {i}")
            with open(mid_embds, "wb") as f:
                pickle.dump(final_embeddings, f)

    # 保存最终结果
    with open(embds_final, "wb") as f:
        pickle.dump(final_embeddings, f)
    print("All AS embeddings generated.")

if __name__ == "__main__":
    main()
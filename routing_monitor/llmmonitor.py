#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ipaddress
# from collections import defaultdict, Counter
# from datetime import timedelta, datetime
from tqdm import tqdm

class Monitor:
    class Node:
        def __init__(self):
            
            self.routes = dict()  # forwarder -> aspath（列表形式）
            self.left = None      # 左子节点
            self.right = None     # 右子节点

        def get_left(self):
            if self.left is None:
                self.left = Monitor.Node()
            return self.left

        def get_right(self):
            if self.right is None:
                self.right = Monitor.Node()
            return self.right

        def find_route(self, forwarder):
            return self.routes.get(forwarder, None)

    def __init__(self):
        self.root = Monitor.Node()
        self.route_changes = []
        self.route_count = 0

    # ---------------------------
    # 树查找与更新，保持原有遍历细节
    # ---------------------------
    # def _find_route_in_tree(self, prefix_str, forwarder):
    #     try:
    #         prefix = ipaddress.ip_network(prefix_str, strict=False)
    #     except Exception:
    #         return None
    #     if prefix.version == 6:
    #         return None
    #     prefixlen = prefix.prefixlen
    #     prefix_int = int(prefix.network_address)
    #     n = self.root
    #     best_match_len = 0
    #     best_candidate = None
    #     for shift in range(prefixlen-1, -1, -1):
    #         left = (prefix_int >> shift) & 1
    #         if left:
    #             n = n.get_left()
    #         else:
    #             n = n.get_right()
    #         candidate = n.find_route(forwarder)
    #         if candidate is not None:
    #             current_match_len = prefixlen - shift
    #             if current_match_len > best_match_len:
    #                 best_match_len = current_match_len
    #                 best_candidate = candidate
    #     if best_candidate is not None:
    #         try:
    #             mask = (1 << 32) - (1 << (32 - best_match_len))
    #             matched_int = prefix_int & mask
    #             matched_prefix = ipaddress.ip_network((matched_int, best_match_len))
    #         except Exception:
    #             matched_prefix = prefix
    #         return [matched_prefix, best_match_len, best_candidate]
    #     else:
    #         return None

    # ---------------------------
    # 树查找：通过转换为32位二进制字符串逐位判断，
    # 当比特位为 '1' 时走左分支，为 '0' 时走右分支
    # ---------------------------
    def _find_route_in_tree(self, prefix_str, forwarder):
        try:
            prefix = ipaddress.ip_network(prefix_str, strict=False)
        except Exception as e:
            print(f"无法解析前缀 {prefix_str}: {e}")
            return None
        if prefix.version == 6:
            return None

        prefix_len = prefix.prefixlen
        # 转换为32位二进制字符串，8位一组，不足的高位补0
        bin_str = '{:032b}'.format(int(prefix.network_address))
        n = self.root
        best_match_len = 0
        best_candidate = None

        # 逐位遍历前缀内的比特位
        for i in range(prefix_len):
            bit = bin_str[i]
            if bit == '1':
                n = n.get_left()
            else:
                n = n.get_right()
            candidate = n.find_route(forwarder)
            if candidate is not None:
                current_match_len = i + 1
                if current_match_len > best_match_len:
                    best_match_len = current_match_len
                    best_candidate = candidate

        if best_candidate is not None:
            # 根据匹配长度构造匹配前缀，后续补0
            matched_bin_str = bin_str[:best_match_len] + '0' * (32 - best_match_len)
            matched_int = int(matched_bin_str, 2)
            try:
                matched_prefix = ipaddress.ip_network((matched_int, best_match_len))
            except Exception as e:
                print(f"计算匹配前缀时出错: {e}")
                matched_prefix = prefix
            # print(f"【查找过程】返回匹配结果: 匹配前缀 = {matched_prefix}, 匹配长度 = {best_match_len}, 路由 AS-PATH = {best_candidate}\n")
            return [matched_prefix, best_match_len, best_candidate]
        else:
            return None

    # ---------------------------
    # 树更新：同样使用32位二进制字符串逐位遍历，写入路由记录到叶节点
    # ---------------------------
    def _update_tree_with_baseline(self, net, forwarder, baseline):
        try:
            net = ipaddress.ip_network(str(net), strict=False)
        except Exception as e:
            print(f"无法解析更新前缀 {net}: {e}")
            return
        prefix_len = net.prefixlen
        bin_str = '{:032b}'.format(int(net.network_address))
        n = self.root
        for i in range(prefix_len):
            bit = bin_str[i]
            if bit == '1':
                n = n.get_left()
            else:
                n = n.get_right()
        n.routes[forwarder] = baseline

    def load_baseline_from_rib(self, df):
        cols = ["timestamp", "prefix", "peer-asn", "as-path"]
        for a in tqdm(df[cols].values, desc="Constructing baseline from RIBS"):
            try:
                net = ipaddress.ip_network(a[1], strict=False)
            except Exception:
                print(f"Warning: invalid prefix {a[1]}")
                continue
            if net.version == 6:
                continue
            aspath = a[3].split(" ")
            forwarder = aspath[0]
            self._update_tree_with_baseline(net, forwarder, aspath)


    def update(self, timestamp, prefix_str, vantage_point, aspath_str, detect):
        try:
            update_net = ipaddress.ip_network(prefix_str, strict=False)
        except Exception:
            return
        if update_net.version == 6:
            return
        update_aspath = aspath_str.split(" ")
        if not update_aspath:
            return
        forwarder = update_aspath[0]
        
        result = self._find_route_in_tree(prefix_str, forwarder)

        if result is not None:
            matched_prefix, match_len, baseline_aspath = result
            stored_prefix = matched_prefix if match_len is not None and match_len != 0 else update_net
            stored_aspath = baseline_aspath
        else:
            stored_prefix, stored_aspath = None, None

        if detect and stored_aspath is not None:
            if prefix_str == '208.65.153.0/24':     # 2008.2.24 - 18:47 UTC
            # if prefix_str == '31.13.67.0/24':     # 2015.6.12 - 8:43 UTC
            # if prefix_str in ['115.116.96.0/24','115.116.97.0/24','115.116.98.0/24', '115.116.99.0/24']:    # 2015.1.7 - 12:00 UTC
            # if prefix_str in ['192.135.33.0/24', '192.84.129.0/24']:    # 2015.1.7 - 9:00 UTC
            # if prefix_str == '101.124.128.0/18':    # 2018.6.29 - 13:00 UTC
            # if prefix_str == '104.18.216.0/21':     # 2020.4.1 - 19:28 UTC
            # if prefix_str == '201.157.24.0/24':     # 2021.2.11 - 4:36 UTC
            # if prefix_str == '24.152.117.0/24':     # 2021.4.16 - 15:07 UTC
            # if prefix_str in ['211.249.221.0/24', '121.53.104.0/24']:    # 2022.2.3 - 01:04 UTC
            # if prefix_str == '44.235.216.0/24':     # 2022.8.17 - 19:39 UTC
            # if prefix_str in ['24.120.56.0/24', '24.120.58.0/24']:  # 2008.8.10 - 19:30 UTC
            # if prefix_str == '1.1.1.1/32':            # 2024.6.27 - 18:51 UTC XXX
            # if prefix_str == '72.20.0.0/24':           # 2016.2.20 - 08:30 UTC
            # if prefix_str in ['205.251.192.0/24', '205.251.193.0/24', '205.251.195.0/24', '205.251.197.0/24', '205.251.199.0/24']: # 2018.4.24 - 11:05 UTC
            # if prefix_str in ['161.123.172.0/24']:      # 2016.04.16 - 07:00 UTC
            # if prefix_str in ['191.86.129.0/24']:         # 2016.05.20 - 21:30 UTC
            # if prefix_str in ['91.108.4.0/24']:           # 2018.07.30 - 06:15 UTC 
                print(f"update {prefix_str} {forwarder} {update_aspath} found {stored_aspath} {stored_prefix}")
            if update_aspath != stored_aspath:
                self.route_changes.append({
                    "timestamp": timestamp,
                    "vantage_point": vantage_point,
                    "forwarder": forwarder,
                    "prefix1": str(stored_prefix),
                    "prefix2": prefix_str,
                    "path1": " ".join(stored_aspath),
                    "path2": " ".join(update_aspath)
                })

        if result is None:
            if prefix_str == '208.65.153.0/24':     # 2008.2.24 - 18:47 UTC
            # if prefix_str == '31.13.67.0/24':     # 2015.6.12 - 8:43 UTC
            # if prefix_str in ['115.116.96.0/24','115.116.97.0/24','115.116.98.0/24', '115.116.99.0/24']:    # 2015.1.7 - 12:00 UTC
            # if prefix_str in ['192.135.33.0/24', '192.84.129.0/24']:    # 2015.1.7 - 9:00 UTC
            # if prefix_str == '101.124.128.0/18':    # 2018.6.29 - 13:00 UTC
            # if prefix_str == '104.18.216.0/21':     # 2020.4.1 - 19:28 UTC
            # if prefix_str == '201.157.24.0/24':     # 2021.2.11 - 4:36 UTC
            # if prefix_str == '24.152.117.0/24':     # 2021.4.16 - 15:07 UTC
            # if prefix_str == '211.249.221.0/24':    # 2022.2.3 - 01:04 UTC
            # if prefix_str == '44.235.216.0/24':     # 2022.8.17 - 19:39 UTC
            # if prefix_str in ['24.120.56.0/24', '24.120.58.0/24']:  # 2008.8.10 - 17:00 UTC
            # if prefix_str == '1.1.1.1/32':            # 2024.6.27 - 18:51 UTC XXX
            # if prefix_str == '72.20.0.0/24':           # 2016.2.20 - 08:30 UTC
            # if prefix_str in ['205.251.192.0/24', '205.251.193.0/24', '205.251.195.0/24', '205.251.197.0/24', '205.251.199.0/24']: # 2018.4.24 - 11:05 UTC
            # if prefix_str in ['161.123.172.0/24']:      # 2016.04.16 - 07:00 UTC
            # if prefix_str in ['191.86.129.0/24']:         # 2016.05.20 - 21:30 UTC
            # if prefix_str in ['91.108.4.0/24']:           # 2018.07.30 - 06:15 UTC 
                print(f"update {prefix_str} {forwarder} NOT found")
            self._update_tree_with_baseline(update_net, forwarder, update_aspath)

    def consume(self, df, detect=False):
        """
        遍历 DataFrame 中的每条记录；如果存在 "A/W" 列，则仅选取 "A" 记录，
        然后依次调用 update() 检测并更新前缀树。
        """
        if "A/W" in df.columns:
            df_A = df.loc[df["A/W"] == "A"]
            # df_W = df.loc[df["A/W"] == "W"]
        # print(f"{len(df_A)} A records, {len(df_W)} W records")
        if len(df_A) == 0:
            print("No A records, loading B records instead")
            df = df.loc[df["A/W"] == "B"]
        else:
            df = df_A
        # print(f"{len(df)} records")
        self.route_count += len(df)
        cols = ["timestamp", "prefix", "peer-asn", "as-path"]
        for a in df[cols].values:
            self.update(*a, detect)

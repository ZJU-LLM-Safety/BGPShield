#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ipaddress
from tqdm import tqdm

class Monitor:
    class Node:
        def __init__(self):
            
            self.routes = dict()  
            self.left = None      
            self.right = None     #

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

    def _find_route_in_tree(self, prefix_str, forwarder):
        try:
            prefix = ipaddress.ip_network(prefix_str, strict=False)
        except Exception as e:
            print(f"Error parsing prefix {prefix_str}: {e}")
            return None
        if prefix.version == 6:
            return None

        prefix_len = prefix.prefixlen
        bin_str = '{:032b}'.format(int(prefix.network_address))
        n = self.root
        best_match_len = 0
        best_candidate = None

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
            matched_bin_str = bin_str[:best_match_len] + '0' * (32 - best_match_len)
            matched_int = int(matched_bin_str, 2)
            try:
                matched_prefix = ipaddress.ip_network((matched_int, best_match_len))
            except Exception as e:
                print(f"Error calculating matched prefix: {e}")
                matched_prefix = prefix
            return [matched_prefix, best_match_len, best_candidate]
        else:
            return None

    def _update_tree_with_baseline(self, net, forwarder, baseline):
        try:
            net = ipaddress.ip_network(str(net), strict=False)
        except Exception as e:
            print(f"Error parsing update prefix {net}: {e}")
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
            self._update_tree_with_baseline(update_net, forwarder, update_aspath)

    def consume(self, df, detect=False):
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

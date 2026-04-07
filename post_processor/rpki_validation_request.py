#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import requests
import json
import time
from pathlib import Path
from requests.exceptions import ConnectionError

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR/"rpki_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

import certifi
def rpki_valid(prefix, asn):
    cache_path = CACHE_DIR/f"{prefix}.{asn}".replace("/", "-")

    try:
        with open(cache_path, "r") as f:
            r = json.load(f)
        return r["data"]["status"]
    except json.JSONDecodeError:
        print(f"[Corrupt cache] {cache_path}, re-querying...")
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass 
    except FileNotFoundError:
        pass

    payload = {"prefix": prefix, "resource": asn}
    url = "https://stat.ripe.net/data/rpki-validation/data.json"
    back_url = "http://stat.ripe.net/data/rpki-validation/data.json"
    # headers = {
    #     "User-Agent": "rpki-client/1.0"
    # }
    try:
        r = requests.get(url, params=payload, timeout=50) #, headers=headers)
        if r.status_code == 200:
            # r = r.json()
            data = r.json()
            if "data" in data and "status" in data["data"]:
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return data["data"]["status"]
            else:
                print(f"[Invalid response format] {data}")
                return "query error"
            # json.dump(r, open(cache_path, "w"))
            # return r["data"]["status"]
        else:
            print(f"RPKI query error: {prefix}, {asn}")
            return "query error"
    except ConnectionError as e:
        time.sleep(1)  
        # print(f"\nHTTPS Connection error: {e}\nTrying HTTP Connection!!!")
        try:
            r = requests.get(back_url, params=payload)
            if r.status_code == 200:
                # r = r.json()
                # json.dump(r, open(cache_path, "w"))
                # return r["data"]["status"]
                data = r.json()
                if "data" in data and "status" in data["data"]:
                    with open(cache_path, "w") as f:
                        json.dump(data, f)
                    print('HTTP Connection Success!!!')
                    return data["data"]["status"]
                else:
                    print(f"[Invalid response format] {data}")
                    return "query error"
            else:
                print(f"RPKI query error: {prefix}, {asn}")
                return "query error"
        except ConnectionError as e:
            # print(f"HTTP Connection error: {e}")
            print(f"RPKI query error {e}: {prefix}, {asn}")
            return "query error"

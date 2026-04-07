"""
!/usr/bin/env python3
-*- coding: utf-8 -*-
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'       
os.environ['MKL_NUM_THREADS'] = '1'       
os.environ['NUMEXPR_NUM_THREADS'] = '1'     
os.environ['OPENBLAS_NUM_THREADS'] = '1'    
os.environ['NUMBA_NUM_THREADS'] = '1'       

from re import T
import threading

threading.current_thread().name = 'MainThread'

import matplotlib
matplotlib.use('Agg')   

from pathlib import Path
from Adapter import BGPShield, force_cleanup
from shutil import get_terminal_size
import click
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.caida_as_rel.fetch_data import get as prepare_edge_file

@click.command()
# @click.option("--serial", "-s", type=click.Choice(["1", "2"]), default="1", help="serial 1 or 2")
@click.option("--time", "-t", type=int, required=True, help="timestamp, e.g., 20200901")
@click.option("--Q", "Q", type=int, default=10, help="hyperparameter Q, e.g., 10")
@click.option("--dimension", type=int, default=16, help="hyperparameter dimension size, e.g., 128")
@click.option("--model", "-M", type=int, default=0, help="id of LLM Model")
@click.option("--epoches", type=int, default=150, help="epoches to train, e.g., 1000")
@click.option("--device", "-d", type=int, default="0", help="Comma-separated GPU ids to use, e.g., '0'")
# @click.option("--num-workers", type=int, default=0, help="number of workers")
def main(time, model, device, **model_params):
    try:
        print(f"Active threads at start: {threading.active_count()}")       
        model_params["time"] = time

        print(f"Using GPU: {device}")
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"

        serial = "1" if time < 20151201 else "2"
        edge_file = prepare_edge_file(serial, time)
        assert edge_file.exists(), f"fail to prepare {edge_file}"


        for k, v in model_params.items():
            print(f"{k}: {v}")
        print("*"*get_terminal_size().columns)

        as_info_path = Path(__file__).resolve().parent/"as_info"/ \
            f"{time}"/f"ases_knowledge_info_base" 
        
        print(f"as_info_path: {as_info_path}")
        as_info_path.mkdir(parents=True, exist_ok=True)
    
        model_list = [
            "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/", 
            "/hub/huggingface/models/Qwen/Qwen3-1.7B-Base/", 
            "/hub/huggingface/models/BAAI/bge-m3/",
            "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
            ]
    
        model_path = model_list[int(model)]

        model_name = model_path.split("/")[-2]
        print(f"Model: {model_name}")

        embed_file = Path(__file__).resolve().parent/f"llMmodels/moreprompt/mean/iterative_as_info/{time}/{model_name}/ases_knowledge_info_base_embd.emb"
        embed_path = os.path.split(embed_file)[0]
        print(f"embed_path: {embed_path}")
        
        train_dir = Path(embed_path) / "L2" / \
            f"{edge_file.stem}.{model_params['epoches']}.{model_params['Q']}.{model_params['dimension']}"

        train_dir.mkdir(parents=True, exist_ok=True)
        print(f"train_dir: {train_dir}")
        
        model_params["train_dir"] = train_dir
        model_params["as_info_path"] = as_info_path
        model_params["embed_file"] = embed_file
        epoches = model_params.pop("epoches")

        print(f"Read edge file from {edge_file}")
        model = BGPShield(**model_params)

        print("Start training...")
        print(f"\tEpoches: {epoches}, Dimension: {model_params['dimension']}, Q: {model_params['Q']}")
        model.train(epoches=epoches)

        print("Save embeddings...")
        model.save_embeddings(path=str(train_dir))
    except Exception as e:
        print(f"Error in main training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Performing final cleanup...")
        force_cleanup()
        print(f"Active threads at end: {threading.active_count()}")       
        print("BGPShield training script completed.")

if __name__ == "__main__":
    main()
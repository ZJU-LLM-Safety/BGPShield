from threadpoolctl import threadpool_limits
threadpool_limits(limits=1, user_api='blas')
threadpool_limits(limits=1, user_api='openmp')

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
import pickle
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Draw
import umap
import hdbscan
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.caida_as_org.fetch_data import build_as_org_snapshot 

def force_cleanup():
    import gc
    
    for _ in range(3):
        gc.collect()
    
    plt.close('all')
    plt.clf()
    plt.cla()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.ipc_collect()
        except:
            pass
    
def worker_init_fn(worker_id):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api='blas')
    threadpool_limits(limits=1, user_api='openmp')

    import torch
    torch.set_num_threads(1)

class Analyzer(Dataset):
    def __init__(self,
                 embd_file,          
                 merged_as_info,     
                 merged_org_info,    
                 Q=10,
                 block_size=256,
                 sample_per_block=1000,
                 device='cuda:0'):
        super().__init__()
        with open(embd_file, 'rb') as f:
            llm_emb = pickle.load(f)
        self.asn_list = list(llm_emb.keys())
        self.role_vectors = np.stack([llm_emb[asn] for asn in self.asn_list], axis=0).astype(np.float32)
        self.hign_embds = self.role_vectors.copy()
        N, _ = self.role_vectors.shape
        self.device = device

        self.merged_as_info = merged_as_info
        self.merged_org_info = merged_org_info

        self.as2org = {asn: self.get_org_name(self.get_org_id(asn)) for asn in self.asn_list}

        self.first_tokens = [
            self.as2org[asn].split()[0].lower()
            for asn in self.asn_list
        ]

        self.Q = Q
        self.block_size = block_size
        self.sample_per_block = sample_per_block

        self.positive_pairs = []
        self.negative_pairs = []

    def get_org_id(self, asn):
        info = self.merged_as_info.get(asn)
        if info:
            return info["opaque_id"] if info["opaque_id"] != "" else info["org_id"]
        return str(asn)

    def get_org_name(self, org_id):
        name = self.merged_org_info.get(org_id, {}).get("name", "").strip()
        if not name:
            name = f"Org_{org_id}"
        return name

    def from_same_org(self, a1, a2):
        return self.as2org[a1] == self.as2org[a2]

    def from_similar_org(self, a1, a2):
        def first_token(asn):
            parts = self.as2org[asn].lower().split()
            return parts[0] if parts else self.as2org[asn].lower()
        return first_token(a1) == first_token(a2)

    def build_sample_pools(self,
                        eps_quantile=0.25,
                        delta_quantile_l=0.75,
                        delta_quantile_h=1.0,
                        max_pos=500_000,
                        neg_pos_ratio=3,
                        ref_block_size=1024,
                        verbose=False):
        import gc
        N, D = self.role_vectors.shape
        emb_cpu = torch.from_numpy(self.role_vectors).float()  # [N, D] on CPU

        sample_vals = []
        try:
            for i in trange(0, N, self.block_size, desc="Stage1 Sampling"):
                row_cpu = emb_cpu[i:i+self.block_size]      # [B, D]
                row_gpu = row_cpu.to(self.device)           # -> GPU

                for j in range(0, N, ref_block_size):
                    col_cpu = emb_cpu[j:j+ref_block_size]
                    col_gpu = col_cpu.to(self.device)

                    with torch.no_grad():
                        dist = ((row_gpu[:,None,:] - col_gpu[None,:,:])**2).sum(-1).sqrt()
                    arr = dist.cpu().numpy().ravel()
                    if arr.size:
                        k = min(self.sample_per_block, arr.size)
                        idx = np.random.choice(arr.size, k, replace=False)
                        sample_vals.append(arr[idx])

                    del col_gpu, col_cpu, dist, arr
                    if j % (ref_block_size * 10) == 0: 
                        torch.cuda.empty_cache()

                del row_gpu, row_cpu

        except Exception as e:
            print(f"Error in Stage 1: {e}")

        if sample_vals:
            samples = np.concatenate(sample_vals)
            eps     = np.quantile(samples, eps_quantile)
            delta_l = np.quantile(samples, delta_quantile_l)
            delta_h = np.quantile(samples, delta_quantile_h)
            del samples
        else:
            eps, delta_l, delta_h = 0.1, 0.5, 1.0
        del sample_vals
        gc.collect()

        pos_reservoir, neg_reservoir = [], []
        pos_seen, neg_seen = 0, 0
        neg_cap = max_pos * neg_pos_ratio

        pos_stats = {"min":1e9,"max":0,"sum":0.0,"count":0}
        neg_stats = {"min":1e9,"max":0,"sum":0.0,"count":0}

        tokens = np.array(self.first_tokens)

        try:
            for i in trange(N, desc="Stage2 Building Pools"):
                row_cpu = emb_cpu[i:i+1]      # [1, D]
                row_gpu = row_cpu.to(self.device)

                drow = np.empty(N, dtype=np.float32)
                for j in range(0, N, ref_block_size):
                    col_cpu = emb_cpu[j:j+ref_block_size]
                    col_gpu = col_cpu.to(self.device)

                    with torch.no_grad():
                        dist_blk = ((row_gpu - col_gpu)**2).sum(-1).sqrt()
                    arr = dist_blk.cpu().numpy().ravel()
                    drow[j:j+arr.shape[0]] = arr

                    del col_gpu, col_cpu, dist_blk, arr
                del row_gpu, row_cpu

                mask_sim = (tokens[i] == tokens)
                pos_indices = np.where(mask_sim & (drow <= eps))[0]

                for j in pos_indices:
                    if i==j: continue
                    v = float(drow[j])
                    pos_stats["min"]  = min(pos_stats["min"],  v)
                    pos_stats["max"]  = max(pos_stats["max"],  v)
                    pos_stats["sum"] += v; pos_stats["count"] += 1
                    pos_seen += 1

                    if len(pos_reservoir) < max_pos:
                        pos_reservoir.append((i,j))
                    else:
                        if random.random() < max_pos / pos_seen:
                            idx_replace = random.randrange(max_pos)
                            pos_reservoir[idx_replace] = (i,j)

                neg_indices = np.where((~mask_sim) & (drow>=delta_l) & (drow<=delta_h))[0]
                for j in neg_indices:
                    if i==j: continue
                    v = float(drow[j])
                    neg_stats["min"]  = min(neg_stats["min"],  v)
                    neg_stats["max"]  = max(neg_stats["max"],  v)
                    neg_stats["sum"] += v; neg_stats["count"] += 1
                    neg_seen += 1

                    if len(neg_reservoir) < neg_cap:
                        neg_reservoir.append((i,j))
                    else:
                        if random.random() < neg_cap / neg_seen:
                            idx_replace = random.randrange(neg_cap)
                            neg_reservoir[idx_replace] = (i,j)

                del drow, mask_sim, pos_indices, neg_indices
                if i % 100 == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in Stage 2: {e}")
        finally:
            del tokens, emb_cpu
            gc.collect()

        random.shuffle(pos_reservoir)
        random.shuffle(neg_reservoir)

        self.positive_pairs = torch.tensor(pos_reservoir, dtype=torch.long)
        self.negative_pairs = torch.tensor(neg_reservoir, dtype=torch.long)

        print(f"[Analyzer] +Pos={len(pos_reservoir):,}, -Neg={len(neg_reservoir):,}")
        print(f"  ↪ Pos dist: min={pos_stats['min']:.4f}, mean={pos_stats['sum']/pos_stats['count']:.4f}, max={pos_stats['max']:.4f}")
        print(f"  ↪ Neg dist: min={neg_stats['min']:.4f}, mean={neg_stats['sum']/neg_stats['count']:.4f}, max={neg_stats['max']:.4f}")
    
        if verbose:
            org_count = {}
            for a1, a2 in pos_reservoir:
                org = self.as2org.get(self.asn_list[a1], 'Unknown')
                org_count[org] = org_count.get(org, 0) + 1
            top_orgs = sorted(org_count.items(), key=lambda x: x[1], reverse=True)[:20]
            print("正样本组织分布:")
            for org, cnt in top_orgs:
                print(f"  {org}: {cnt}")

            print("正样本组织示例:")
            for a1, a2 in pos_reservoir[:20]:
                distance = np.sqrt(((self.role_vectors[a1] - self.role_vectors[a2])**2).sum())
                print(f"  {self.asn_list[a1]} ↔ {self.asn_list[a2]} ({self.as2org[self.asn_list[a1]]}) distance={distance:.4f}")

            print("负样本组织示例:")
            for a1, a2 in neg_reservoir[:20]:
                distance = np.sqrt(((self.role_vectors[a1] - self.role_vectors[a2])**2).sum())
                print(f"  {self.asn_list[a1]} ↛ {self.asn_list[a2]} ({self.as2org[self.asn_list[a1]]} vs {self.as2org[self.asn_list[a2]]}) distance={distance:.4f}")

    def plot_eval_sample_pairs(self, pos_idx, neg_idx, epoch=0, save_path=None, draw_cluster_ellipses=True):
        try:
            emb = torch.from_numpy(self.role_vectors).cpu().float()
            proj = TSNE(n_components=2, random_state=42).fit_transform(emb)
            # proj = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(emb.numpy())
            del emb

            asn_subset = self.asn_list
            orgs = [self.as2org.get(asn, 'Unknown') for asn in asn_subset]
            unique_orgs = sorted(set(orgs))
            org2brightness = {org: i / max(1, len(unique_orgs) - 1) for i, org in enumerate(unique_orgs)}
            brightness = [org2brightness[org] for org in orgs]
            gray_colors = [(b, b, b) for b in brightness]

            fig, ax = plt.subplots(figsize=(10, 10))

            asn2coord = {asn: proj[i] for i, asn in enumerate(asn_subset)}
            asn2idx = {asn: i for i, asn in enumerate(asn_subset)}

            ax.scatter(proj[:, 0], proj[:, 1], c=gray_colors, s=12, alpha=0.5, zorder=1)

            highlighted_pos = set()
            highlighted_neg = set()

            n_pos = self.positive_pairs.size(0)
            k_pos = min(100, n_pos)

            # pos_sample_idx = torch.randperm(n_pos, device='cpu')[:k_pos]
            pos_sample_idx = random.sample(range(n_pos), k_pos)

            pos_idx_set = set(pos_idx)
            for idx in pos_sample_idx:
                a, b = self.positive_pairs[idx].tolist()
                a1, a2 = self.asn_list[a], self.asn_list[b]
                # if a in pos_idx_set and b in pos_idx_set:
                if a1 in asn2coord and a2 in asn2coord:
                    x1, y1 = asn2coord[a1]
                    x2, y2 = asn2coord[a2]
                    ax.plot([x1, x2], [y1, y2], color='green', alpha=0.6, linewidth=1.0, zorder=2)
                    highlighted_pos.update([a1, a2])

            n_neg = self.negative_pairs.size(0)
            k_neg = min(100, n_neg)
            
            # neg_sample_idx = torch.randperm(n_neg, device='cpu')[:k_neg]
            neg_sample_idx = random.sample(range(n_neg), k_neg)

            neg_idx_set = set(neg_idx)
            for idx in neg_sample_idx:
                a, b = self.negative_pairs[idx].tolist()
                a1, a2 = self.asn_list[a], self.asn_list[b]
                # if a in neg_idx_set and b in neg_idx_set:
                if a1 in asn2coord and a2 in asn2coord:
                    x1, y1 = asn2coord[a1]
                    x2, y2 = asn2coord[a2]
                    ax.plot([x1, x2], [y1, y2], color='red', alpha=0.6, linewidth=1.0, zorder=2)
                    highlighted_neg.update([a1, a2])

            coords_pos = [asn2coord[asn] for asn in highlighted_pos if asn in asn2coord]
            if coords_pos:
                coords_pos = np.array(coords_pos)
                ax.scatter(coords_pos[:, 0], coords_pos[:, 1],
                        c='limegreen', s=18, alpha=0.9, edgecolors='black', linewidths=0.5, zorder=3)

            coords_neg = [asn2coord[asn] for asn in highlighted_neg if asn in asn2coord]
            if coords_neg:
                coords_neg = np.array(coords_neg)
                ax.scatter(coords_neg[:, 0], coords_neg[:, 1],
                        c='orangered', s=18, alpha=0.9, edgecolors='black', linewidths=0.5, zorder=3)

            ax.set_title(f"[Epoch {epoch}]: Sample Pairs after Dim Reducer")

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            else:
                plt.show()
            
            del proj, asn_subset, orgs, unique_orgs, org2brightness, brightness, gray_colors
            del asn2coord, asn2idx, highlighted_pos, highlighted_neg, coords_pos, coords_neg
        except Exception as e:
            print(f"Error in plotting: {e}")

        finally:
            plt.close(fig)
            plt.clf()
            plt.cla()

    def __len__(self):
        # return len(self.positive_pairs)
        return self.positive_pairs.size(0) if len(self.positive_pairs) > 0 else 0

    
    def __getitem__(self, idx):
        pi, pj = self.positive_pairs[idx].tolist()

        Q = self.Q
        Nneg = self.negative_pairs.size(0)

        # neg_idx = torch.randint(0, Nneg, (Q,), dtype=torch.long)
        neg_idx = random.sample(range(Nneg), min(Q, Nneg))

        neg_sel = self.negative_pairs[neg_idx]
        neg_i = neg_sel[:, 0].tolist()
        neg_j = neg_sel[:, 1].tolist()

        return pi, pj, neg_i, neg_j
    
class EnhancedEmbeddingReducer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[None, None], dropout=0.1):
        super().__init__()
        
        if hidden_dims[0] is None:
            hidden_dims[0] = min(input_dim, max(output_dim * 4, input_dim // 2))
        if hidden_dims[1] is None:
            hidden_dims[1] = max(output_dim * 2, hidden_dims[0] // 2)
            
        self.input_projection = None
        
        self.input_projection = nn.Linear(input_dim, output_dim)
            
        layers = []
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        layers.extend([
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout)
        ])
        
        layers.append(nn.Linear(hidden_dims[1], output_dim))
        self.net = nn.Sequential(*layers)
        
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        output = self.net(x)
        
        attention_weights = self.attention(output)
        output = output * attention_weights
        
        output = output + self.input_projection(x)
            
        return output

def collate_fn(batch):
    if not batch:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long) 
        
    pos_i = torch.tensor([item[0] for item in batch], dtype=torch.long)
    pos_j = torch.tensor([item[1] for item in batch], dtype=torch.long)
    neg_i = torch.tensor([item[2] for item in batch], dtype=torch.long)  # [B, Q]
    neg_j = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return pos_i, pos_j, neg_i, neg_j

class BGPShield:
    def __init__(self,
                 time,
                 bge=False,
                 Q = 10,
                 dimension = 128,
                 train_dir = Path("./"),
                 as_info_path = Path("./"),
                 embed_file = Path("./"),
                 device = 'cuda:0'
                 ):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        merged_as_info, merged_org_info = build_as_org_snapshot(time, window=10)
        self.as_info_path = as_info_path
        self.bge = bge

        self.train_dir = Path(train_dir)
        self.train_dir.mkdir(exist_ok=True, parents=True)
        
        self.analyzer = Analyzer(
            embd_file=embed_file,
            merged_as_info=merged_as_info,
            merged_org_info=merged_org_info,
            Q=Q,
            # eps_quantile=0.25,
            # delta_quantile_l=0.7,
            # delta_quantile_h=0.95,
            block_size=256,
            device=self.device,
        )

        # Reducer & Optimizer
        D_high = self.analyzer.role_vectors.shape[1]
        # self.reducer = EmbeddingReducer(D_high, dimension).to(self.device)
        self.reducer = EnhancedEmbeddingReducer(D_high, dimension, dropout=0.1).to(self.device)
        self.optimizer = optim.AdamW(self.reducer.parameters(), lr=1e-4, weight_decay=0.01 * 1e-4)
        self.device = torch.device(self.device)
        self.loss_log = []
        # self.num_workers = num_workers

        if (self.train_dir/"reducer_final.pt").exists():
            print("Model already trained")
            self.reducer.load_state_dict(torch.load(self.train_dir/"reducer_final.pt", map_location=self.device))
            return 

        self.checkpoint_path = self.train_dir / "checkpoint.pth"
        self.start_epoch = 1
        if self.checkpoint_path.exists():
            cp = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            print(f"Loading checkpoint from epoch {cp['epoch']}")
            self.reducer.load_state_dict(cp['model_state_dict'])
            self.optimizer.load_state_dict(cp['optimizer_state_dict'])
            self.analyzer.role_vectors = cp['role_vectors']
            # self.analyzer.build_sample_pools()
            self.loss_log = cp.get('loss_log', [])
            self.start_epoch = cp['epoch'] + 1
        self.analyzer.build_sample_pools(eps_quantile=0.25, delta_quantile_l=0.75, delta_quantile_h=1, verbose=True)

        n_total_pos = self.analyzer.positive_pairs.size(0)
        n_total_neg = self.analyzer.negative_pairs.size(0)

        n_vis_pos = min(80, n_total_pos)
        n_vis_neg = min(100, n_total_neg)

        # self.vis_pos_idx = torch.randperm(n_total_pos)[:n_vis_pos]
        # self.vis_neg_idx = torch.randperm(n_total_neg)[:n_vis_neg]

        self.vis_pos_idx = random.sample(range(n_total_pos), n_vis_pos)
        self.vis_neg_idx = random.sample(range(n_total_neg), n_vis_neg)


    def train(self,
              epoches=150,
              batch_size=1024,
              refresh_interval=25,
              log_interval=10,     
              save_interval=20,     
              loader_interval=5,   
              pos_sample_size=50000,  
              ):
        if (self.train_dir/"reducer_final.pt").exists():
            self.reducer.load_state_dict(torch.load(self.train_dir/"reducer_final.pt", map_location=self.device))
            return

        log_name = "{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_dir = self.train_dir / "logs" / log_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)

        emb_high = None  
        total_pos = len(self.analyzer)
        selected_idx = random.sample(range(total_pos), min(pos_sample_size, total_pos))

        sampler = torch.utils.data.SubsetRandomSampler(selected_idx)

        loader = DataLoader(
            self.analyzer,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=False,
        )

        print(f"Start training at epoch {self.start_epoch}")
        for epoch in range(self.start_epoch, epoches+1):
            begin_time = time.time()
            if emb_high is None:
                emb_high = torch.from_numpy(self.analyzer.hign_embds).to(self.device)
            
            if epoch % refresh_interval == 0:
                with torch.no_grad():
                    emb_low = self.reducer(emb_high).cpu().numpy()
                self.analyzer.role_vectors = emb_low.copy()
                tmp_dp = (0.75 - 0.65) / epoches
                tmp_eps_quantile = 0.25 + tmp_dp * (epoch - 1)
                tmp_delta_quantile_l = 0.75 - tmp_dp * (epoch - 1)
                tmp_delta_quantile_h = 1 - tmp_dp * (epoch - 1)
                self.analyzer.build_sample_pools(eps_quantile=tmp_eps_quantile, delta_quantile_l=tmp_delta_quantile_l, delta_quantile_h=tmp_delta_quantile_h)

                del emb_low
                
            if epoch % loader_interval == 0 and epoch != epoches:
                total_pos = len(self.analyzer)
                selected_idx = random.sample(range(total_pos), min(pos_sample_size, total_pos))

                sampler = torch.utils.data.SubsetRandomSampler(selected_idx)

                loader = DataLoader(
                    self.analyzer,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=0,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    persistent_workers=False,
                )

            self.reducer.train()
            total_loss = 0.0

            for pos_i, pos_j, neg_i, neg_j in loader:
                pos_i = pos_i.to(self.device)
                pos_j = pos_j.to(self.device)
                neg_i = neg_i.to(self.device)
                neg_j = neg_j.to(self.device)

                emb_pos_i = self.reducer(emb_high[pos_i])
                emb_pos_j = self.reducer(emb_high[pos_j])

                B, Q_ = neg_i.shape
                emb_neg_i = self.reducer(emb_high[neg_i.view(-1)]).view(B, Q_, -1)
                emb_neg_j = self.reducer(emb_high[neg_j.view(-1)]).view(B, Q_, -1)

                if not self.bge:
                    pos_dist = ((emb_pos_i - emb_pos_j) ** 2).sum(dim=1).sqrt()
                    neg_dist = ((emb_neg_i - emb_neg_j) ** 2).sum(dim=2).sqrt()
                else:
                    pos_dist = - np.log(emb_pos_i @ emb_pos_j.T + 1e-6)
                    neg_dist = - torch.log(torch.bmm(emb_neg_i, emb_neg_j.permute(0,2,1)) + 1e-6)

                loss = F.softplus(pos_dist.unsqueeze(1) - neg_dist).mean()

                del emb_pos_i, emb_pos_j, emb_neg_i, emb_neg_j, pos_dist, neg_dist

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            self.loss_log.append(avg_loss)
            writer.add_scalar('Loss/train', avg_loss, epoch)
            end_time = time.time()
            elapsed_time = end_time - begin_time
            print(f"[Epoch {epoch}] \tTime = {elapsed_time:.2f}s, \tLoss = {avg_loss:16.8e}")

            if epoch % save_interval == 0 or epoch == epoches:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.reducer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'role_vectors': self.analyzer.role_vectors,
                    'loss_log': self.loss_log
                }, self.checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch}")

            if epoch % 11 == 0:
                force_cleanup()

        writer.close()

        torch.save(self.reducer.state_dict(), self.train_dir/"reducer_final.pt")
        print("Training complete. Final model saved.")

    def save_embeddings(self, path='.'):
        print("save embeddings...")
        embds_file = os.path.join(path, "ases_knowledge_info_base_embd.emb")
        if os.path.exists(embds_file):
            os.remove(embds_file)

        with torch.no_grad():
            emb_high = torch.from_numpy(self.analyzer.hign_embds).to(self.device)
            final_emb = self.reducer(emb_high).cpu().numpy()
        
        embds = dict(zip(self.analyzer.asn_list, final_emb))
        with open(embds_file, 'wb') as f:
            pickle.dump(embds, f)
        print("save embeddings done")

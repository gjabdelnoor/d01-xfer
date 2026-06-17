#!/usr/bin/env python
"""
EVO2-7B block-26 embeddings + Goodfire L26 SAE top-K for the D0.1 RdRp panel.
Full-GPU (RTX 3090, 24 GB) — FP8 disabled for Ampere (sm_86), reusing the
proven config-swap from evo2_backend.py. Resumable via a memmap + progress file.

Outputs (in --out dir):
  evo2_block26_mean.npy   float32 [N, 4096]   mean-pooled block-26 embedding
  evo2_meta.csv           accession,length_nt_used,forward_s
  evo2_sae_top64.csv      accession,rank,feat_id,act     (long format)
  progress.json           {"done": k}                    resume marker
"""
import os, sys, time, json, argparse, types
import numpy as np
import torch

# --- Ampere (sm_86) has no flash-attn build here: stub the compiled ext so
#     vortex imports cleanly, and we force the SDPA attention path below. ---
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sae", default="")           # path to SAE .pt (optional)
ap.add_argument("--layer", default="blocks.26.mlp.l3")
ap.add_argument("--maxlen", type=int, default=8192)
ap.add_argument("--k", type=int, default=64)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---- FP8 disable for Ampere (lifted from evo2_backend.py) ------------------
import evo2.utils as _u
_cfgdir = os.path.join(os.path.dirname(_u.__file__), "configs")
_nofp8 = os.path.join(_cfgdir, "evo2-7b-1m.no-fp8.yml")
if "no-fp8" not in _u.CONFIG_MAP.get("evo2_7b", ""):
    if not os.path.exists(_nofp8):
        import yaml
        src = os.path.join(_cfgdir, "evo2-7b-1m.yml")
        cfg = yaml.safe_load(open(src))
        cfg["use_fp8_input_projections"] = False  # sm_86 has no FP8
        cfg["use_flash_attn"] = False             # no flash build -> SDPA path
        cfg["use_flash_rmsnorm"] = False
        yaml.safe_dump(cfg, open(_nofp8, "w"))
    _u.CONFIG_MAP["evo2_7b"] = "configs/evo2-7b-1m.no-fp8.yml"
    print(f"[embed] FP8 disabled -> {_u.CONFIG_MAP['evo2_7b']}", flush=True)

from evo2 import Evo2
torch.set_grad_enabled(False)
print("[embed] loading evo2_7b ...", flush=True)
t0 = time.time()
model = Evo2("evo2_7b")
tok = model.tokenizer
print(f"[embed] model loaded in {time.time()-t0:.0f}s", flush=True)

# ---- optional SAE ----------------------------------------------------------
W = b_enc = b_dec = None
if args.sae and os.path.exists(args.sae):
    sd = torch.load(args.sae, map_location="cuda:0", weights_only=False)
    W = sd["_orig_mod.W"].to("cuda:0", torch.float32)
    b_enc = sd["_orig_mod.b_enc"].to("cuda:0", torch.float32)
    b_dec = sd["_orig_mod.b_dec"].to("cuda:0", torch.float32)
    print(f"[embed] SAE loaded W={tuple(W.shape)} K={args.k}", flush=True)

def sae_topk(hmean):  # hmean: [4096] cpu float
    x = hmean.to("cuda:0", torch.float32) - b_dec
    pre = x @ W + b_enc
    v, i = pre.topk(args.k)
    return i.cpu().numpy(), v.relu().cpu().numpy()

# ---- fasta -----------------------------------------------------------------
def read_fasta(p):
    name, seq = None, []
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(seq)
            name = line[1:].split()[0]; seq = []
        else:
            seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)

records = list(read_fasta(args.fasta))
N = len(records)
DIM = 4096
print(f"[embed] {N} sequences", flush=True)

emb_path = os.path.join(args.out, "evo2_block26_mean.npy")
# memmap so a crash keeps finished rows
if os.path.exists(emb_path):
    embs = np.lib.format.open_memmap(emb_path, mode="r+")
else:
    embs = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, DIM))

prog_path = os.path.join(args.out, "progress.json")
start = 0
if os.path.exists(prog_path):
    start = json.load(open(prog_path)).get("done", 0)
    print(f"[embed] resuming at {start}", flush=True)

meta_f = open(os.path.join(args.out, "evo2_meta.csv"), "a")
if start == 0:
    meta_f.write("accession,length_nt_used,forward_s\n")
sae_f = open(os.path.join(args.out, "evo2_sae_top64.csv"), "a")
if start == 0 and W is not None:
    sae_f.write("accession,rank,feat_id,act\n")

for j in range(start, N):
    acc, seq = records[j]
    s = seq.upper().replace("U", "T")
    cap = args.maxlen
    while True:
        ss = s[:cap]
        try:
            ids = torch.tensor(tok.tokenize(ss), dtype=torch.int64).unsqueeze(0).to("cuda:0")
            t1 = time.time()
            out, emb = model(ids, return_embeddings=True, layer_names=[args.layer])
            h = emb[args.layer]
            if isinstance(h, tuple):
                h = h[0]
            hmean = h[0].float().mean(0).cpu()  # [4096]
            dt = time.time() - t1
            del ids, out, emb, h
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cap <= 1024:
                raise
            cap //= 2
            print(f"[embed] OOM {acc} len={len(s)} -> retry cap={cap}", flush=True)
    embs[j] = hmean.numpy()
    meta_f.write(f"{acc},{len(ss)},{dt:.2f}\n")
    if (j + 1) % 50 == 0:
        torch.cuda.empty_cache()
    if W is not None:
        fi, fv = sae_topk(hmean)
        for r in range(len(fi)):
            sae_f.write(f"{acc},{r+1},{int(fi[r])},{float(fv[r]):.4f}\n")
    if (j + 1) % 20 == 0 or j == N - 1:
        embs.flush(); meta_f.flush(); sae_f.flush()
        json.dump({"done": j + 1}, open(prog_path, "w"))
        eta = (N - j - 1) * dt / 60.0
        print(f"[embed] {j+1}/{N}  last={dt:.2f}s  eta~{eta:.0f}min", flush=True)

embs.flush(); json.dump({"done": N}, open(prog_path, "w"))
print("[embed] DONE", flush=True)

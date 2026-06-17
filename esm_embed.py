#!/usr/bin/env python
"""
ESM2-3B (esm2_t36_3B_UR50D) mean-pooled embeddings for the D0.1 RdRp panel.
Input is the nucleotide RdRp CDS fasta; we translate best-of-3-forward-frames
to protein, then embed. Runs on the vast 3090 (fp16). Resumable.

Outputs (in --out):
  esm2_3b_mean.npy   float32 [N, 2560]
  esm_meta.csv       accession,aa_len_used,frame,n_internal_stops,forward_s
  progress.json
"""
import os, sys, time, json, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", required=True)         # nucleotide RdRp CDS
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="facebook/esm2_t36_3B_UR50D")
ap.add_argument("--max_aa", type=int, default=1022)   # ESM2 trained context
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---- standard genetic code -------------------------------------------------
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a+b+c: _AAS[i*16+j*4+k]
         for i, a in enumerate(_BASES) for j, b in enumerate(_BASES) for k, c in enumerate(_BASES)}

def translate(nt, frame):
    nt = nt[frame:]
    nt = nt[: len(nt) - (len(nt) % 3)]
    return "".join(CODON.get(nt[i:i+3], "X") for i in range(0, len(nt), 3))

def best_protein(nt):
    nt = nt.upper().replace("U", "T")
    best = None
    for f in (0, 1, 2):
        aa = translate(nt, f)
        core = aa[:-1] if aa.endswith("*") else aa
        stops = core.count("*")
        # score: fewer internal stops, then longer
        key = (stops, -len(core))
        if best is None or key < best[0]:
            best = (key, aa, f, stops)
    _, aa, frame, stops = best
    aa = aa.replace("*", "")           # drop stops for the LM
    return aa, frame, stops

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
print(f"[esm] {N} sequences", flush=True)

from transformers import AutoTokenizer, EsmModel
tok = AutoTokenizer.from_pretrained(args.model)
print("[esm] loading model ...", flush=True)
t0 = time.time()
model = EsmModel.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
DIM = model.config.hidden_size
print(f"[esm] loaded {args.model} dim={DIM} in {time.time()-t0:.0f}s", flush=True)

emb_path = os.path.join(args.out, "esm2_3b_mean.npy")
if os.path.exists(emb_path):
    embs = np.lib.format.open_memmap(emb_path, mode="r+")
else:
    embs = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, DIM))

prog_path = os.path.join(args.out, "progress.json")
start = json.load(open(prog_path)).get("done", 0) if os.path.exists(prog_path) else 0
if start:
    print(f"[esm] resuming at {start}", flush=True)

meta = open(os.path.join(args.out, "esm_meta.csv"), "a")
if start == 0:
    meta.write("accession,aa_len_used,frame,n_internal_stops,forward_s\n")

for j in range(start, N):
    acc, nt = records[j]
    aa, frame, stops = best_protein(nt)
    aa = aa[: args.max_aa]
    if len(aa) == 0:
        aa = "M"
    enc = tok(aa, return_tensors="pt", truncation=True, max_length=args.max_aa + 2)
    t1 = time.time()
    with torch.no_grad():
        out = model(**{k: v.to("cuda") for k, v in enc.items()})
    last = out.last_hidden_state[0].float()        # [L, DIM]
    L = int(enc["attention_mask"][0].sum())
    vec = last[1:L-1].mean(0).cpu().numpy()         # drop CLS/EOS, mean over residues
    dt = time.time() - t1
    embs[j] = vec
    meta.write(f"{acc},{len(aa)},{frame},{stops},{dt:.2f}\n")
    del out, last
    if (j + 1) % 50 == 0 or j == N - 1:
        embs.flush(); meta.flush()
        json.dump({"done": j + 1}, open(prog_path, "w"))
        torch.cuda.empty_cache()
        print(f"[esm] {j+1}/{N} last={dt:.2f}s", flush=True)

embs.flush(); json.dump({"done": N}, open(prog_path, "w"))
print("[esm] DONE", flush=True)

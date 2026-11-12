"""Jukebox layer-66 features for EDGE on a small GPU (tested target: 8 GB VRAM, 8 GB RAM).

jukemirlib needs >=13 GB VRAM and loads the whole 10 GB prior into RAM. This
does the same computation, but streams the transformer one layer at a time from
the shards that shard_jukebox_prior.py writes. Each layer is put on the GPU, every
audio slice runs through it, and it is freed before the next one loads. The math
copies jukemirlib's get_activations_custom (fp32 activations, fp16 conv weights
as stored), so the features match what EDGE was trained on.

Writes test.py's --use_cached_features layout:
  <out>/<song>/<song>_sliceN.wav  and  <song>_sliceN.npy   (each .npy is (150, 4800))

Usage:
  python data/audio_extraction/jukebox_lowmem.py --music_dir custom_music --out cached_features
  python test.py --use_cached_features --feature_cache_dir cached_features --save_motions ...
"""

import argparse
import glob
import os
import sys
from functools import cmp_to_key
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # EDGE repo root

import jukemirlib.lib as jl  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from jukebox.hparams import Hyperparams, setup_hparams  # noqa: E402
from jukebox.make_models import MODELS, make_prior, make_vqvae  # noqa: E402
from jukemirlib.setup_models import load_weights, set_module_tensor_to_device  # noqa: E402

from data.slice import slice_audio  # noqa: E402

LAYER = 66  # what EDGE's jukebox_features.py extracts
FPS = 30
CACHE = os.path.expanduser("~/.cache/jukemirlib")
DEVICE = "cuda"


def slice_key(path):
  return int(os.path.splitext(path)[0].split("slice")[-1])


def build_models(shard_dir):
  """Same model construction as jukemirlib.setup_models, without loading the transformer layers."""
  # jukebox calls dist.get_rank() whenever torch.distributed is available, which
  # newer torch rejects without a process group, so start a 1-process group.
  import torch.distributed as dist
  if dist.is_available() and not dist.is_initialized():
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29513", rank=0, world_size=1)

  VQVAE, *priors = MODELS["5b"]
  with init_empty_weights():
    vqvae = make_vqvae(setup_hparams(VQVAE, dict(sample_length=1048576, restore_vqvae="")), "meta")
  with init_empty_weights():
    prior = make_prior(setup_hparams(priors[-1], dict(restore_prior="")), vqvae, "meta")
  prior.prior.only_encode = True

  load_weights(vqvae, os.path.join(CACHE, "vqvae.pth.tar"), DEVICE)
  base = torch.load(os.path.join(shard_dir, "base.pt"), map_location="cpu")
  for k, v in base.items():
    set_module_tensor_to_device(prior, k, DEVICE, value=v)
  del base
  return vqvae, prior


def layer_tensors(shard_dir, i):
  return torch.load(os.path.join(shard_dir, f"layer_{i:02d}.pt"), map_location="cpu")


@torch.no_grad()
def extract_song(wav_files, vqvae, prior, shard_dir, chunk):
  jl.DEVICE = DEVICE  # get_z / get_cond read the device from jukemirlib
  import jukemirlib
  jukemirlib.DEVICE = DEVICE

  # 1. VQ-VAE tokens, one slice at a time (all slices are the same length).
  zs = []
  for f in tqdm(wav_files, desc="VQ-VAE"):
    zs.append(jl.get_z(vqvae, jl.load_audio(f)).cpu())
  z = torch.cat(zs)  # (N, D)
  N = z.shape[0]

  # 2. Conditioning + input embedding (jukemirlib get_cond / get_activations_custom preamble).
  x_cond, y_cond = jl.get_cond(prior)
  pr = prior.prior
  xs = []
  for s in range(0, N, chunk):
    zc = z[s:s + chunk].to(DEVICE)[:, :jl.T]
    n, D = zc.shape
    x = pr.preprocess(zc)
    x = pr.x_emb(x)
    x = jl.roll(x, 1)
    x[:, 0] = y_cond.repeat(n, 1, 1).view(n, pr.width)
    x = pr.x_emb_dropout(x) + pr.pos_emb_dropout(pr.pos_emb())[:D] + x_cond.repeat(n, 1, 1)[:, :D]
    xs.append(x.float().cpu())
  x = torch.cat(xs)  # (N, D, 4800), kept on CPU between layers

  # 3. Stream layers 1..LAYER.
  layers = pr.transformer._attn_mods
  for i in tqdm(range(LAYER), desc="Jukebox layers"):
    tensors = layer_tensors(shard_dir, i)
    for k, v in tensors.items():
      set_module_tensor_to_device(prior, k, DEVICE, value=v)
    l = layers[i]
    for s in range(0, N, chunk):
      xc = x[s:s + chunk].to(DEVICE)
      l.attn.del_cache()
      xc = l(xc, encoder_kv=None, sample=True)
      l.attn.del_cache()
      x[s:s + chunk] = xc.cpu()
    for k in tensors:  # free the layer
      set_module_tensor_to_device(prior, k, "meta")
    del tensors
    torch.cuda.empty_cache()

  # 4. Downsample each slice to 30 fps, like jukemirlib.extract(downsample_target_rate=30).
  return [jl.downsample(np.array(x[n]), target_rate=FPS).astype(np.float32) for n in range(N)]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--music_dir", default="custom_music")
  p.add_argument("--out", default="cached_features")
  p.add_argument("--shards", default=os.path.join(CACHE, "prior_shards"))
  p.add_argument("--chunk", type=int, default=4, help="slices per GPU batch")
  a = p.parse_args()

  wavs = sorted(glob.glob(os.path.join(a.music_dir, "*.wav")))
  assert wavs, f"no .wav in {a.music_dir}"
  vqvae, prior = build_models(a.shards)
  for wav in wavs:
    song = Path(wav).stem
    d = os.path.join(a.out, song)
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"Slicing {wav}")
    slice_audio(wav, 2.5, 5.0, d)
    files = sorted(glob.glob(os.path.join(d, "*.wav")), key=slice_key)
    feats = extract_song(files, vqvae, prior, a.shards, a.chunk)
    for f, rep in zip(files, feats):
      np.save(os.path.splitext(f)[0] + ".npy", rep)
    print(f"{song}: {len(files)} slices, features {feats[0].shape}")


if __name__ == "__main__":
  main()

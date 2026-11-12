"""Split the Jukebox 5B top-prior checkpoint into one file per transformer layer.

The checkpoint (prior_level_2.pth.tar, 10.3 GB) uses torch's legacy (pre-zip)
format, so torch.load would pull the whole file into RAM, and it can't be
memory-mapped. This streams it instead: the pickle is parsed with placeholder
storages, then the raw storage bytes, which follow the pickle in order, are read
one at a time. Each layer's tensors are written out as soon as that layer is
complete, so peak RAM is a few hundred MB.

Output (default ~/.cache/jukemirlib/prior_shards/):
  base.pt       every model tensor outside the transformer layers
  layer_XX.pt   prior.transformer._attn_mods.XX.*   (XX = 0..71)

Usage: python data/audio_extraction/shard_jukebox_prior.py [--ckpt PATH] [--out DIR]
"""

import argparse
import os
import pickle
import re
import struct
from collections import defaultdict

import torch

LAYER_RE = re.compile(r"^prior\.transformer\._attn_mods\.(\d+)\.")


class _Storage:
  def __init__(self, key, dtype, numel):
    self.key, self.dtype, self.numel = key, dtype, numel


class _TensorRef:
  def __init__(self, storage, offset, size, stride):
    self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
  return _TensorRef(storage, offset, size, stride)


class _LazyUnpickler(pickle.Unpickler):
  def __init__(self, f):
    super().__init__(f, encoding="latin1")
    self.storages = {}

  def find_class(self, module, name):
    if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
      return _rebuild_tensor_v2
    if module == "torch" and name.endswith("Storage"):
      return getattr(torch, name)
    return super().find_class(module, name)

  def persistent_load(self, saved_id):
    typename, storage_type, key, location, numel, view_metadata = saved_id
    assert typename == "storage" and view_metadata is None, saved_id
    if key not in self.storages:
      self.storages[key] = _Storage(key, storage_type.dtype, numel)
    return self.storages[key]


def shard(ckpt_path, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  with open(ckpt_path, "rb") as f:
    for _ in range(3):  # magic number, protocol version, sys info
      pickle.load(f)
    up = _LazyUnpickler(f)
    obj = up.load()
    storage_keys = pickle.load(f)

    state = obj["model"]
    group_of = {}  # tensor name -> group
    by_storage = defaultdict(list)  # storage key -> [tensor names]
    remaining = defaultdict(int)  # group -> tensors not yet read
    for name, ref in state.items():
      m = LAYER_RE.match(name)
      g = f"layer_{int(m.group(1)):02d}" if m else "base"
      group_of[name] = g
      by_storage[ref.storage.key].append(name)
      remaining[g] += 1
    print(f"{len(state)} tensors in {len(remaining)} groups, {len(storage_keys)} storages")

    done = {g for g in remaining if os.path.exists(os.path.join(out_dir, g + ".pt"))}
    buffers = defaultdict(dict)
    for i, key in enumerate(storage_keys):
      st = up.storages[key]
      (numel,) = struct.unpack("<q", f.read(8))
      assert numel == st.numel, (key, numel, st.numel)
      nbytes = numel * torch.empty((), dtype=st.dtype).element_size()
      names = by_storage.get(key, [])
      if not names or all(group_of[n] in done for n in names):
        f.seek(nbytes, os.SEEK_CUR)  # optimizer state etc., or already sharded
        continue
      flat = torch.frombuffer(bytearray(f.read(nbytes)), dtype=st.dtype)
      for n in names:
        ref = state[n]
        t = flat.as_strided(ref.size, ref.stride, ref.offset).clone()
        g = group_of[n]
        buffers[g][n] = t
        remaining[g] -= 1
        if remaining[g] == 0 and g not in done:
          torch.save(buffers.pop(g), os.path.join(out_dir, g + ".pt"))
          done.add(g)
          print(f"  wrote {g}  ({i + 1}/{len(storage_keys)} storages read)", flush=True)
    assert all(v == 0 for v in remaining.values()), "some tensors were never read"
  print("done")


if __name__ == "__main__":
  cache = os.path.expanduser("~/.cache/jukemirlib")
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", default=os.path.join(cache, "prior_level_2.pth.tar"))
  p.add_argument("--out", default=os.path.join(cache, "prior_shards"))
  a = p.parse_args()
  shard(a.ckpt, a.out)

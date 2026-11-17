"""Music beats with librosa, on the motion's frame grid. Runs in EDGE's venv (it has librosa).

  .venv/bin/python policy_eval/music_beats.py OUT.json --fps 50 --n_frames 1499 \
      --song custom_music/funkorama.wav --offset 7.5 --hop 512 --tightness 100 \
      --extra librosa:vibeace --extra self:offset=0.0

Method (MINT tools/preprocessing.py, with the motion's fps in place of AIST++'s 60):
  sr = fps * hop  so one onset-envelope frame == one motion frame
  y  = librosa.load(song, sr=sr, offset=offset, duration=n_frames / fps)
  env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
  start_bpm = librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop)[0]
      (MINT reads the BPM from the AIST++ file name; there is none here, so librosa estimates it)
  beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop,
                                  start_bpm=start_bpm, tightness=tightness)
Beat indices are motion frame indices (0 = motion frame 0 = `offset` seconds into the song).
"""

import argparse
import json
import sys

import librosa
import numpy as np


def beats_for(path, offset, fps, n_frames, hop, tightness):
  sr = int(round(fps * hop))
  dur = n_frames / fps
  y, _ = librosa.load(path, sr=sr, offset=offset, duration=dur, mono=True)
  if len(y) < int(dur * sr) - hop:
    raise ValueError(f"{path}: only {len(y) / sr:.2f} s of audio after offset {offset} s, "
                     f"need {dur:.2f} s")
  env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
  start_bpm = float(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop)[0])
  tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop,
                                         start_bpm=start_bpm, tightness=tightness)
  beats = np.asarray(beats, dtype=int)
  beats = beats[(beats >= 0) & (beats < n_frames)]
  return {"path": str(path), "offset_s": offset, "sr": sr, "hop": hop,
          "start_bpm": start_bpm, "tempo_bpm": float(np.atleast_1d(tempo)[0]),
          "beat_frames": beats.tolist(), "n_beats": int(len(beats)),
          "onset_env_len": int(len(env))}


def resolve(spec, song, song_offset):
  """'librosa:key' -> example track at 0 s; 'self:offset=S' -> the song at S s."""
  if spec.startswith("librosa:"):
    return librosa.ex(spec.split(":", 1)[1]), 0.0
  if spec.startswith("self:offset="):
    return song, float(spec.split("=", 1)[1])
  raise ValueError(f"unknown calibration song spec {spec!r}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--song", required=True)
  ap.add_argument("--offset", type=float, required=True)
  ap.add_argument("--fps", type=float, required=True)
  ap.add_argument("--n_frames", type=int, required=True)
  ap.add_argument("--hop", type=int, default=512)
  ap.add_argument("--tightness", type=float, default=100.0)
  ap.add_argument("--extra", action="append", default=[])
  a = ap.parse_args()
  out = {"librosa_version": librosa.__version__,
         "song": beats_for(a.song, a.offset, a.fps, a.n_frames, a.hop, a.tightness),
         "calibration": {}}
  for spec in a.extra:
    path, off = resolve(spec, a.song, a.offset)
    out["calibration"][spec] = beats_for(path, off, a.fps, a.n_frames, a.hop, a.tightness)
  with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
  print(f"music beats -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
  main()

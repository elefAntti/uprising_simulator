
from __future__ import annotations
from typing import Optional, Dict, Any
import os, json, numpy as np

def _as1d(x, name="array"):
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 0: a = a.reshape(1)
    elif a.ndim > 1: a = a.reshape(-1)
    return a

class NPZShardLogger:
    def __init__(self, out_dir: str, shard_size:int=10000, prefix:str="distill_run", compress:bool=True):
        self.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        self.shard_size = int(shard_size); self.prefix = prefix; self.compress = compress
        self._buf = []; self._shard_idx = 0; self._obs_dim = None; self._act_dim = None

    def _flush(self):
        if not self._buf: return None
        name = f"{self.prefix}_{self._shard_idx:05d}.npz"; path = os.path.join(self.out_dir, name)
        self._shard_idx += 1

        N = len(self._buf)
        D = int(self._buf[0]["obs_flat"].shape[0])
        A = int(self._buf[0]["action"].shape[0])

        obs = np.zeros((N, D), dtype="float32")
        action = np.zeros((N, A), dtype="float32")
        reward = np.zeros((N,), dtype="float32")
        done = np.zeros((N,), dtype="bool")
        ep = np.zeros((N,), dtype="int32")
        tt = np.zeros((N,), dtype="int32")
        teacher = np.empty((N,), dtype="object")

        # next_obs handling: not every row has it -> make dense array + mask
        any_next = any(b["next_obs_flat"] is not None for b in self._buf)
        if any_next:
            next_obs = np.zeros((N, D), dtype="float32")
            has_next = np.zeros((N,), dtype="bool")
        else:
            next_obs = None
            has_next = None

        for i, b in enumerate(self._buf):
            obs[i] = b["obs_flat"]
            action[i] = b["action"]
            reward[i] = float(b.get("reward", 0.0))
            done[i] = bool(b.get("done", False))
            ep[i] = int(b.get("episode_id", 0))
            tt[i] = int(b.get("t", 0))
            teacher[i] = str(b.get("teacher","")).encode("utf-8")
            if next_obs is not None and b["next_obs_flat"] is not None:
                # if lengths mismatch (shouldn't), we clip/pad
                n = b["next_obs_flat"]
                nd = min(D, n.shape[0])
                next_obs[i, :nd] = n[:nd]
                if nd < D:  # pad stays zero
                    pass
                has_next[i] = True

        save_fn = np.savez_compressed if self.compress else np.savez
        if next_obs is None:
            save_fn(path, obs=obs, action=action, reward=reward, done=done, episode_id=ep, t=tt, teacher=teacher)
        else:
            save_fn(path, obs=obs, next_obs=next_obs, has_next=has_next, action=action, reward=reward, done=done, episode_id=ep, t=tt, teacher=teacher)

        self._buf.clear(); return path

    def log_step(self, episode_id:int, t:int, obs_flat, action, done:bool, reward:float=0.0, teacher:str="", next_obs_flat=None, extras:Optional[Dict[str,Any]]=None):
        obs_arr = _as1d(obs_flat, "obs_flat")
        act_arr = _as1d(action, "action")
        next_arr = None if next_obs_flat is None else _as1d(next_obs_flat, "next_obs_flat")
        if self._obs_dim is None:
            self._obs_dim = int(obs_arr.shape[0]); self._act_dim = int(act_arr.shape[0])
        self._buf.append({
            "episode_id": episode_id, "t": t, "obs_flat": obs_arr, "next_obs_flat": next_arr,
            "action": act_arr, "reward": float(reward), "done": bool(done), "teacher": teacher, "extras": extras or {}
        })
        if len(self._buf) >= self.shard_size: return self._flush()
        return None

    def close(self): return self._flush()

class ParquetLogger:
    def __init__(self, path: str, schema: Optional[dict] = None):
        try:
            import pyarrow as pa, pyarrow.parquet as pq  # noqa: F401
        except Exception as e:
            raise RuntimeError("ParquetLogger requires 'pyarrow'. pip install pyarrow") from e
        self.pa = pa; self.pq = pq; self.path = path; self._writer = None; self._schema = None

    def log_step(self, episode_id:int, t:int, obs_flat, action, done:bool, reward:float=0.0, teacher:str="", next_obs_flat=None, extras:Optional[Dict[str,Any]]=None):
        pa, pq = self.pa, self.pq
        obs_arr = _as1d(obs_flat, "obs_flat")
        act_arr = _as1d(action, "action")
        next_arr = _as1d(next_obs_flat, "next_obs_flat") if next_obs_flat is not None else np.array([], dtype=np.float32)

        row = {
            "episode_id": pa.scalar(episode_id, type=pa.int32()), "t": pa.scalar(t, type=pa.int32()),
            "teacher": pa.scalar(teacher, type=pa.string()), "reward": pa.scalar(float(reward), type=pa.float32()),
            "done": pa.scalar(bool(done), type=pa.bool_()),
            "action_left": pa.scalar(float(act_arr[0]), type=pa.float32()),
            "action_right": pa.scalar(float(act_arr[1]) if act_arr.shape[0]>1 else float(act_arr[0]), type=pa.float32()),
            "obs_flat": pa.array(list(map(float, obs_arr)), type=pa.list_(pa.float32())),
            "next_obs_flat": pa.array(list(map(float, next_arr)), type=pa.list_(pa.float32())),
            "has_next": pa.scalar(bool(next_obs_flat is not None), type=pa.bool_())
        }
        if extras: row["extras_json"] = pa.scalar(json.dumps(extras), type=pa.string())
        batch = pa.record_batch(row)
        if self._writer is None:
            self._schema = batch.schema; self._writer = pq.ParquetWriter(self.path, self._schema)
        self._writer.write_table(pa.Table.from_batches([batch]))

    def close(self):
        if self._writer is not None: self._writer.close(); self._writer=None

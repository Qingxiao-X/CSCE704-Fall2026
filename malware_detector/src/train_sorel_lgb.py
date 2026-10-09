"""Train the large LightGBM ensemble member on EMBER 2018 + a SOREL-20M sample.

SOREL-20M (Sophos/ReversingLabs) publishes EMBER feature_version=2 vectors
for ~20M PE files (first seen 2017-01 .. 2019-04) in a public S3 bucket:

    https://sorel-20m.s3.amazonaws.com/09-DEC-2020/processed-data/meta.db                 (3.8 GB, SQLite)
    https://sorel-20m.s3.amazonaws.com/09-DEC-2020/processed-data/ember_features/data.mdb (77 GB, LMDB)
    https://sorel-20m.s3.amazonaws.com/09-DEC-2020/processed-data/ember_features/lock.mdb

Steps (each skipped if its output already exists):
  1. sample N benign + N malicious SOREL rows first seen on/after --min-date,
     excluding every sha256 in the EMBER 2018 test set (52,300 of its files are
     also in SOREL), and dump their vectors to a float32 memmap;
  2. bin EMBER's labeled training rows + the SOREL sample into a LightGBM
     binary Dataset, streaming from the memmaps (RAM-safe on a 4 GB machine);
  3. train with Elastic's EMBER benchmark parameters, logging AUC and TPR at
     0.5%/1% FPR on the EMBER test set every 50 rounds.

    python -m src.train_sorel_lgb --sorel-dir data/sorel --out data/lgb/runC.txt
"""

from __future__ import annotations

import argparse
import datetime
import os
import sqlite3
import time
import zlib

import numpy as np

from src.features import FEATURE_DIM
from src.train_ember import read_vectorized

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=2048, max_depth=15, min_data_in_leaf=50,
    feature_fraction=0.5, num_threads=2, verbose=-1, seed=42, force_col_wise=True,
    # Small enough for a 4 GB machine: bin boundaries from 40k sampled rows,
    # histogram cache capped (2048-leaf trees would otherwise cache GBs).
    bin_construct_sample_cnt=40000, histogram_pool_size=200,
)


def ember_test_sha256(ember_raw_dir: str) -> set:
    import re

    pat = re.compile(r'"sha256":\s*"([0-9a-f]{64})"')
    with open(os.path.join(ember_raw_dir, "test_features.jsonl")) as f:
        return {pat.search(line).group(1) for line in f if line.strip()}


def extract_sorel(sorel_dir, prefix, n_per_class, min_ts, exclude, seed=7) -> int:
    import lmdb
    import msgpack

    db = sqlite3.connect(f"file:{os.path.join(sorel_dir, 'meta.db')}?mode=ro", uri=True)
    rng = np.random.default_rng(seed)
    rows = []
    for label in (0, 1):
        cand = [r for r in db.execute(
            "select sha256, is_malware from meta where is_malware=? and rl_fs_t>=?", (label, min_ts)
        ) if r[0] not in exclude]
        pick = rng.choice(len(cand), size=min(n_per_class, len(cand)), replace=False)
        rows += [cand[i] for i in pick]
        print(f"[sorel] class {label}: {len(cand)} available, {len(pick)} sampled", flush=True)
    rows.sort()  # key order == LMDB order, so reads are mostly sequential

    env = lmdb.open(sorel_dir, readonly=True, lock=False, map_size=int(1e12), readahead=False)
    X = np.memmap(prefix + "_X.dat", dtype=np.float32, mode="w+", shape=(len(rows), FEATURE_DIM))
    y = np.zeros(len(rows), np.float32)
    k = 0
    with env.begin() as txn:
        for sha, label in rows:
            v = txn.get(sha.encode())
            if v is None:
                continue
            X[k] = np.asarray(msgpack.loads(zlib.decompress(v), strict_map_key=False)[0], dtype=np.float32)
            y[k] = label
            k += 1
            if k % 100000 == 0:
                print(f"[sorel] {k}/{len(rows)}", flush=True)
    X.flush()
    np.save(prefix + "_y.npy", y[:k])
    with open(prefix + "_n.txt", "w") as f:
        f.write(str(k))
    return k


def _sequence(X, idx):
    import lightgbm as lgb

    class Seq(lgb.Sequence):
        batch_size = 4096

        def __getitem__(self, i):
            if isinstance(i, slice):
                ii = idx[i]
                if len(ii) == 0:
                    return np.empty((0, X.shape[1]), np.float64)
                return np.asarray(X[ii[0] : ii[-1] + 1])[ii - ii[0]].astype(np.float64)
            return np.asarray(X[idx[i]], dtype=np.float64)

        def __len__(self):
            return len(idx)

    return Seq()


def main(argv=None) -> int:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, roc_curve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sorel-dir", default="data/sorel")
    ap.add_argument("--ember-raw-dir", default="data/ember2018_raw/ember2018")
    ap.add_argument("--ember-vectorized-dir", default="data/ember2018_vectorized")
    ap.add_argument("--work-dir", default="data/lgb")
    ap.add_argument("--n-per-class", type=int, default=450000)
    ap.add_argument("--min-date", default="2018-01-01")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--out", default="data/lgb/runC.txt")
    args = ap.parse_args(argv)
    os.makedirs(args.work_dir, exist_ok=True)

    prefix = os.path.join(args.work_dir, "sorel_sample")
    if not os.path.exists(prefix + "_n.txt"):
        min_ts = datetime.datetime.fromisoformat(args.min_date).replace(tzinfo=datetime.timezone.utc).timestamp()
        extract_sorel(args.sorel_dir, prefix, args.n_per_class, min_ts, ember_test_sha256(args.ember_raw_dir))
    with open(prefix + "_n.txt") as f:
        n_sorel = int(f.read())
    X_s = np.memmap(prefix + "_X.dat", dtype=np.float32, mode="r", shape=(n_sorel, FEATURE_DIM))
    y_s = np.load(prefix + "_y.npy")

    X_e, y_e, X_t, y_t = read_vectorized(args.ember_vectorized_dir, FEATURE_DIM)
    y_e, y_t = np.asarray(y_e), np.asarray(y_t)
    labeled = np.where(y_e != -1)[0]

    train_bin = os.path.join(args.work_dir, "ember_sorel_train.bin")
    test_bin = os.path.join(args.work_dir, "ember_test.bin")
    ds_params = {"max_bin": 255, "verbose": -1, "bin_construct_sample_cnt": PARAMS["bin_construct_sample_cnt"]}
    if not os.path.exists(train_bin):
        d = lgb.Dataset([_sequence(X_e, labeled), _sequence(X_s, np.arange(n_sorel))],
                        label=np.concatenate([y_e[labeled], y_s]), params=ds_params)
        d.construct()
        d.save_binary(train_bin)
        lgb.Dataset(_sequence(X_t, np.arange(len(y_t))), label=y_t, reference=d, params=ds_params).construct().save_binary(test_bin)
    dtrain = lgb.Dataset(train_bin)
    dtest = lgb.Dataset(test_bin, reference=dtrain)

    def tpr_at(y, p, fpr):
        f, t, _ = roc_curve(y, p)
        return t[np.searchsorted(f, fpr, side="right") - 1]

    def feval(p, ds):
        y = ds.get_label()
        return [("auc", roc_auc_score(y, p), True), ("tpr@0.5%", tpr_at(y, p, 0.005), True), ("tpr@1%", tpr_at(y, p, 0.01), True)]

    t0 = time.time()

    def log(env):
        if (env.iteration + 1) % 50 == 0:
            print(env.iteration + 1, f"{time.time() - t0:.0f}s", [(r[1], round(r[2], 5)) for r in env.evaluation_result_list], flush=True)

    booster = lgb.train(PARAMS, dtrain, num_boost_round=args.rounds, valid_sets=[dtest], feval=feval, callbacks=[log])
    booster.save_model(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

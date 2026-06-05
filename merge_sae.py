#!/usr/bin/env python3
"""
merge_sae.py
------------
PHASE 4 merge: fold the SAE score arrays from extract_sae.py
(sae_l{N}_{variant}.npz) into the master atlas SQLite as a `sae_features` table.

    python merge_sae.py --variant l0_50
    python merge_sae.py --variant l0_50 --min-activation 0.001   # prune dead features
    python merge_sae.py --variant l0_50 --gold                   # print surgical-gold targets

Schema (mirrors bouncer_features, minus `component` — SAEs are residual-stream):
    sae_features(layer, variant, feature_idx,
                 topic_fstat, bouncer_fstat, bouncer_delta,
                 mean_corp, mean_auth, activation_rate, corp_leaning)

Surgical gold = high bouncer_fstat + low topic_fstat = corporate-discriminating
but NOT capability-specific → safest to ablate. Each feature's decoder vector
W_dec[:,feature_idx] in the Qwen-Scope SAE is the steering knob.
"""
from __future__ import annotations
import argparse, sqlite3, glob, re
from pathlib import Path
import numpy as np

ATLAS = Path("atlas")

DDL = """
CREATE TABLE IF NOT EXISTS sae_features (
    layer           INTEGER,
    variant         TEXT,
    feature_idx     INTEGER,
    topic_fstat     REAL,
    bouncer_fstat   REAL,
    bouncer_delta   REAL,
    mean_corp       REAL,
    mean_auth       REAL,
    activation_rate REAL,
    corp_leaning    INTEGER,
    PRIMARY KEY (layer, variant, feature_idx)
);
CREATE INDEX IF NOT EXISTS idx_sae_bouncer ON sae_features(variant, bouncer_fstat DESC);
CREATE INDEX IF NOT EXISTS idx_sae_topic   ON sae_features(variant, topic_fstat);
"""

GOLD_SQL = """
SELECT layer, feature_idx,
       ROUND(bouncer_fstat,1) AS bouncer_F,
       ROUND(topic_fstat,2)   AS topic_F,
       ROUND(bouncer_delta,3) AS delta,
       ROUND(activation_rate,4) AS act_rate
FROM sae_features
WHERE variant = ?
  AND bouncer_fstat IS NOT NULL
  AND bouncer_fstat > ?
  AND topic_fstat  < ?
ORDER BY bouncer_fstat DESC
LIMIT ?;
"""


def merge(atlas: Path, variant: str, min_act: float) -> int:
    db = atlas / "atlas.sqlite"
    if not db.exists():
        raise SystemExit(f"atlas sqlite not found: {db}  (run Phase 3 build first)")
    files = sorted(glob.glob(f"sae_l*_{variant}.npz"),
                   key=lambda p: int(re.search(r"sae_l(\d+)_", p).group(1)))
    if not files:
        raise SystemExit(f"no sae_l*_{variant}.npz files here — run extract_sae.py first")

    con = sqlite3.connect(db)
    con.executescript(DDL)
    con.execute("DELETE FROM sae_features WHERE variant = ?", (variant,))   # idempotent re-merge

    total = 0
    for f in files:
        L = int(re.search(r"sae_l(\d+)_", f).group(1))
        z = np.load(f, allow_pickle=True)
        topic = z["topic_fstat"]; act = z["activation_rate"]
        has_b = "bouncer_fstat" in z.files
        bouncer = z["bouncer_fstat"] if has_b else np.full_like(topic, np.nan)
        delta   = z["bouncer_delta"] if has_b else np.full_like(topic, np.nan)
        mcorp   = z["mean_corp"]     if has_b else np.full_like(topic, np.nan)
        mauth   = z["mean_auth"]     if has_b else np.full_like(topic, np.nan)

        keep = act > min_act                                    # drop dead features
        idx = np.nonzero(keep)[0]
        rows = [(int(L), variant, int(i), float(topic[i]),
                 None if not has_b else float(bouncer[i]),
                 None if not has_b else float(delta[i]),
                 None if not has_b else float(mcorp[i]),
                 None if not has_b else float(mauth[i]),
                 float(act[i]),
                 None if not has_b else int(delta[i] > 0)) for i in idx]
        con.executemany(
            "INSERT OR REPLACE INTO sae_features VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        total += len(rows)
        print(f"  layer {L:>2}: {len(rows):>6} features kept (of {len(topic)})")
    con.commit()
    print(f"merged {total} sae_features rows  (variant={variant})  → {db}")
    return total


def show_gold(atlas: Path, variant: str, min_bouncer: float, max_topic: float, n: int):
    con = sqlite3.connect(atlas / "atlas.sqlite")
    rows = con.execute(GOLD_SQL, (variant, min_bouncer, max_topic, n)).fetchall()
    if not rows:
        print("no surgical-gold features matched (need a bouncer pass + tune thresholds)"); return
    print(f"\n  SURGICAL GOLD — high bouncer_F (>{min_bouncer}) + low topic_F (<{max_topic})")
    print(f"  {'layer':>5} {'feat':>6} {'bouncer_F':>10} {'topic_F':>8} {'delta':>7} {'act':>7}")
    for L, fi, bF, tF, d, a in rows:
        lean = "corp" if (d or 0) > 0 else "auth"
        print(f"  {L:>5} {fi:>6} {bF:>10} {tF:>8} {d:>7} {a:>7}   ({lean})")
    print("\n  → steer/ablate with the SAE decoder vector W_dec[:, feat] at that layer.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="l0_50", choices=["l0_50", "l0_100"])
    p.add_argument("--atlas", default=str(ATLAS))
    p.add_argument("--min-activation", type=float, default=0.0,
                   help="drop features that fire on <= this fraction of tokens")
    p.add_argument("--gold", action="store_true", help="just print surgical-gold targets")
    p.add_argument("--min-bouncer", type=float, default=10.0)
    p.add_argument("--max-topic", type=float, default=3.0)
    p.add_argument("--n", type=int, default=25)
    a = p.parse_args()
    atlas = Path(a.atlas).expanduser()

    if a.gold:
        show_gold(atlas, a.variant, a.min_bouncer, a.max_topic, a.n)
    else:
        merge(atlas, a.variant, a.min_activation)
        show_gold(atlas, a.variant, a.min_bouncer, a.max_topic, a.n)


if __name__ == "__main__":
    main()

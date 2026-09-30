"""Build a zone-subset NYC dataset.

The full build has 62 in-region zones, which is ~4x the compute the study can
afford. This filters the existing window parquets down to the N busiest taxi
zones so the environment, the demand and the fleet stay consistent.

Zone selection uses TRAIN windows only: picking zones by val/test volume would
leak evaluation information into the environment definition.

    uv run python scripts/build_zone_subset.py --num-zones 16

Writes data/nyc{N}/splits/{train,val,test}/*.parquet + manifest.json and
data/nyc{N}/zone_ids.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/nyc")
BBOX = (-74.02, 40.70, -73.93, 40.80)


def nearest_zone(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-zones", type=int, default=16)
    ap.add_argument("--src", type=str, default=str(SRC))
    args = ap.parse_args()

    src = Path(args.src)
    cen = pd.read_csv(src / "zone_centroids.csv")
    lon, lat = cen["lon"].to_numpy(), cen["lat"].to_numpy()
    in_bbox = (
        (lon >= BBOX[0]) & (lon <= BBOX[2])
        & (lat >= BBOX[1]) & (lat <= BBOX[3])
    )
    in_region = np.where(in_bbox)[0]
    pts_all = cen[["lon", "lat"]].to_numpy()

    manifest = json.loads((src / "splits" / "manifest.json").read_text())
    train_files = [src / "splits" / e["file"]
                   for e in manifest["splits"]["train"]]

    counts = np.zeros(len(cen), dtype=np.int64)
    rows = []
    for f in train_files:
        df = pd.read_parquet(f)
        z = nearest_zone(df[["origin_x", "origin_y"]].to_numpy(), pts_all)
        counts += np.bincount(z, minlength=len(cen))
        rows.append(len(df))

    # Rank only the in-region zones; ties broken by LocationID for determinism.
    ids = cen["LocationID"].to_numpy()
    order = sorted(in_region, key=lambda i: (-counts[i], ids[i]))
    keep_idx = sorted(order[: args.num_zones], key=lambda i: ids[i])
    keep_ids = sorted(int(ids[i]) for i in keep_idx)

    total = int(counts.sum())
    kept_share = 100.0 * counts[keep_idx].sum() / max(total, 1)
    print(f"in-region zones: {len(in_region)} -> keeping {len(keep_idx)}")
    print(f"train order volume kept: {kept_share:.1f}% of {total}")
    print(f"zone ids (LocationID): {keep_ids}")

    out_root = Path(f"data/nyc{args.num_zones}")
    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    keep_mask = np.zeros(len(cen), dtype=bool)
    keep_mask[keep_idx] = True

    new_manifest = {"bbox": list(BBOX), "window_min": manifest.get("window_min", 60),
                    "sample_rate": manifest.get("sample_rate", 1.0),
                    "source": str(src), "num_zones": len(keep_idx),
                    "zone_ids": keep_ids, "splits": {}}

    for split in ("train", "val", "test"):
        (out_root / "splits" / split).mkdir(parents=True, exist_ok=True)
        entries = []
        kept_counts = []
        for e in manifest["splits"][split]:
            df = pd.read_parquet(src / "splits" / e["file"])
            z = nearest_zone(df[["origin_x", "origin_y"]].to_numpy(), pts_all)
            sub = df[keep_mask[z]]
            name = f"{split}/window_{len(entries):04d}.parquet"
            sub.to_parquet(out_root / "splits" / name, index=False)
            entries.append({**e, "file": name})
            kept_counts.append(len(sub))
        new_manifest["splits"][split] = entries
        print(f"{split}: {len(entries)} windows, "
              f"orders/window min={min(kept_counts)} max={max(kept_counts)} "
              f"mean={sum(kept_counts)/len(kept_counts):.0f}")

    (out_root / "splits" / "manifest.json").write_text(json.dumps(new_manifest, indent=2))
    (out_root / "zone_ids.json").write_text(json.dumps(keep_ids, indent=2))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()

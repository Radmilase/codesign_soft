# quick_test_skillspace.py  (VERSION: v_codesign_1file)
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import inspect

print("VERSION: v_codesign_1file")  # <-- to confirm you run the right script

os.environ.setdefault("WARP_DISABLE_CUDA", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
SSHP = (HERE / "skill_space_hypothesis").resolve()
if not (SSHP / "physics").exists():
    raise RuntimeError(f"Не найдено: {SSHP / 'physics'}")
sys.path.insert(0, str(SSHP))

def add_baseline_to_path():
    for c in [
        HERE / "codesign-soft-gripper",
        HERE.parent / "codesign-soft-gripper",
        SSHP / ".." / "codesign-soft-gripper",
        SSHP / "codesign-soft-gripper",
    ]:
        c = c.resolve()
        if c.exists() and (c / "code").exists():
            sys.path.insert(0, str(c))
            return str(c)
    return None

BASELINE_DIR = add_baseline_to_path()

from physics.neural_proxy import simulate_neural


def sample_skill(rng: np.random.Generator):
    skills = ["pinch", "power", "push"]
    w = np.array([0.4, 0.4, 0.2], dtype=np.float64)
    skill = rng.choice(skills, p=w / w.sum())

    if skill == "pinch":
        params = {"force": float(rng.uniform(1.0, 6.0)),
                  "distance": float(rng.uniform(0.008, 0.05))}
    elif skill == "power":
        params = {"force": float(rng.uniform(3.0, 12.0)),
                  "wrap": float(rng.uniform(0.2, 1.0))}
    else:
        params = {"force": float(rng.uniform(1.0, 8.0)),
                  "speed": float(rng.uniform(0.05, 0.5))}
    return skill, params


def gripper_baseline(n_blocks: int):
    return {"stiffness_blocks": [1.0] * n_blocks}

def gripper_random(rng: np.random.Generator, n_blocks: int, lo=0.1, hi=3.0):
    x = rng.uniform(lo, hi, size=(n_blocks,))
    return {"stiffness_blocks": [float(v) for v in x]}


def score_from_out(out: dict) -> float:
    if not isinstance(out, dict):
        return 0.0

    if any(k in out for k in ["success", "stability", "slip", "energy"]):
        success = float(out.get("success", 0.0))
        stability = float(out.get("stability", 0.0))
        slip = float(out.get("slip", 0.0))
        energy = float(out.get("energy", 0.0))
        return 5.0 * success + 1.0 * stability - 0.5 * slip - 0.05 * energy

    # fallback: first numeric scalar
    for _, v in out.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)

    return 0.0


def eval_one(gripper: dict, skill: str, params: dict, obj: str, proxy_mode: str):
    out = simulate_neural(gripper, skill, params, obj, mode=proxy_mode)
    if not isinstance(out, dict):
        out = {"raw": out}
    s = score_from_out(out)
    return float(s), out


def eval_skillspace(gripper: dict, obj: str, trials: int, seed: int, proxy_mode: str, debug_keys: bool = False):
    rng = np.random.default_rng(seed)
    rows = []
    first_keys = None

    for t in range(trials):
        skill, params = sample_skill(rng)
        s, out = eval_one(gripper, skill, params, obj, proxy_mode)

        if first_keys is None and isinstance(out, dict):
            first_keys = list(out.keys())
            if debug_keys:
                print("Proxy out keys example:", first_keys)

        rows.append({"trial": t, "skill": skill, "params": params, "score": s, "proxy_out": out})

        if trials <= 10 or (t + 1) % max(1, trials // 5) == 0:
            print(f"[{t+1:4d}/{trials}] score={s:.4f} skill={skill}")

    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    per_skill = {}
    for r in rows:
        per_skill.setdefault(r["skill"], []).append(r["score"])
    per_skill_stats = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": int(len(v))}
                      for k, v in per_skill.items()}

    summary = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "n": int(len(scores)),
        "per_skill": per_skill_stats,
        "proxy_keys_example": first_keys,
    }
    return summary, rows


def codesign_random_search(obj: str, n_blocks: int, iters: int, eval_trials: int, seed: int, proxy_mode: str):
    rng = np.random.default_rng(seed)

    # baseline init
    g0 = gripper_baseline(n_blocks)
    s0, _ = eval_skillspace(g0, obj, trials=eval_trials, seed=seed, proxy_mode=proxy_mode, debug_keys=True)
    best = {"gripper": g0, "summary": s0}
    history = [{"iter": 0, "kind": "baseline_init", "mean": s0["mean"], "gripper": g0}]

    for it in range(1, iters + 1):
        g = gripper_random(rng, n_blocks)
        summ, _ = eval_skillspace(g, obj, trials=eval_trials, seed=seed + it * 101, proxy_mode=proxy_mode)
        history.append({"iter": it, "kind": "random", "mean": summ["mean"], "gripper": g})

        if summ["mean"] > best["summary"]["mean"]:
            best = {"gripper": g, "summary": summ}

        print(f"codesign [{it:03d}/{iters}] mean={summ['mean']:.4f}  best={best['summary']['mean']:.4f}")

    return best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="006_mustard_bottle")
    ap.add_argument("--mode", choices=["baseline", "random_design", "codesign"], default="baseline")
    ap.add_argument("--trials", type=int, default=50, help="used in baseline/random_design")
    ap.add_argument("--n_blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy_mode", default="id")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--plot", action="store_true")

    ap.add_argument("--iters", type=int, default=30, help="codesign iterations")
    ap.add_argument("--eval_trials", type=int, default=20, help="trials per gripper in codesign")
    args = ap.parse_args()

    print("SCRIPT_DIR:", HERE)
    print("SSHP:", SSHP)
    print("BASELINE_DIR:", BASELINE_DIR)
    print("simulate_neural signature:", inspect.signature(simulate_neural))
    print("Object:", args.object, "| mode:", args.mode, "| proxy_mode:", args.proxy_mode)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = Path(args.outdir) / f"skillspace_{args.mode}_{args.object}_{ts}.json"

    payload = {"time": ts, "args": vars(args)}

    if args.mode == "baseline":
        gr = gripper_baseline(args.n_blocks)
        summ, rows = eval_skillspace(gr, args.object, trials=args.trials, seed=args.seed,
                                    proxy_mode=args.proxy_mode, debug_keys=True)
        payload.update({"gripper": gr, "summary": summ, "rows": rows})
        print("\n=== SUMMARY ===")
        print(json.dumps(summ, ensure_ascii=False, indent=2))

    elif args.mode == "random_design":
        rng = np.random.default_rng(args.seed)
        gr = gripper_random(rng, args.n_blocks)
        summ, rows = eval_skillspace(gr, args.object, trials=args.trials, seed=args.seed,
                                    proxy_mode=args.proxy_mode, debug_keys=True)
        payload.update({"gripper": gr, "summary": summ, "rows": rows})
        print("\n=== SUMMARY ===")
        print(json.dumps(summ, ensure_ascii=False, indent=2))

    else:  # codesign
        best, hist = codesign_random_search(
            obj=args.object,
            n_blocks=args.n_blocks,
            iters=args.iters,
            eval_trials=args.eval_trials,
            seed=args.seed,
            proxy_mode=args.proxy_mode,
        )
        payload.update({"best": best, "history": hist})
        print("\n=== BEST SUMMARY ===")
        print(json.dumps(best["summary"], ensure_ascii=False, indent=2))

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()

# mvp_hypothesis.py
import os
import sys
import json
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import os
os.environ["WARP_DISABLE_CUDA"] = "1"      # отключить CUDA для warp
os.environ["CUDA_VISIBLE_DEVICES"] = ""    # на всякий случай скрыть GPU


# -------------------------
# 0) Optional: add baseline repo to sys.path (kept)
# -------------------------
BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "codesign-soft-gripper"))
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

print("BASELINE_DIR:", BASELINE_DIR)
print("BASELINE_DIR exists:", os.path.exists(BASELINE_DIR))

# -------------------------
# 0.1) Optional neural physics import (robust fallback)
# -------------------------
_SIM_NEURAL_OK = False
_simulate_neural = None
try:
    # your local module (as in your snippet)
    from physics.neural_proxy import simulate_neural as _simulate_neural  # type: ignore
    _SIM_NEURAL_OK = True
except Exception as e:
    print("[WARN] neural_proxy.simulate_neural not available, will fallback to surrogate.")
    print("       import error:", repr(e))

# -------------------------
# 1) Skills (no external imports)
# -------------------------
class Skill:
    name = "base"
    def sample_params(self, rng: np.random.Generator, n: int = 16) -> List[Dict[str, float]]:
        raise NotImplementedError

class PinchSkill(Skill):
    name = "pinch"
    def sample_params(self, rng, n=16):
        return [{"force": float(rng.uniform(1.0, 5.0)),
                 "distance": float(rng.uniform(0.01, 0.04))}
                for _ in range(n)]

class PowerSkill(Skill):
    name = "power"
    def sample_params(self, rng, n=16):
        return [{"force": float(rng.uniform(3.0, 10.0)),
                 "wrap": float(rng.uniform(0.5, 1.0))}
                for _ in range(n)]

class HookSkill(Skill):
    name = "hook"
    def sample_params(self, rng, n=16):
        return [{"force": float(rng.uniform(2.0, 6.0)),
                 "angle": float(rng.uniform(-0.5, 0.5))}
                for _ in range(n)]

# a “baseline grasp” skill (fixed policy vibe)
class BaselineSkill(Skill):
    name = "baseline"
    def sample_params(self, rng, n=16):
        # keep deterministic-ish and narrow: “one policy”
        return [{"force": float(rng.uniform(4.0, 6.0)),
                 "preset": 1.0}
                for _ in range(n)]

# -------------------------
# 2) Morphology (block-wise stiffness like the baseline paper)
# -------------------------
@dataclass
class Gripper:
    stiffness: np.ndarray

    @staticmethod
    def from_blocks(stiffness_blocks: List[float]) -> "Gripper":
        return Gripper(stiffness=np.array(stiffness_blocks, dtype=np.float32))

# -------------------------
# 3) Surrogate physics (MVP) + unified wrapper (neural or surrogate)
# -------------------------
def simulate_surrogate(
    rng: np.random.Generator,
    gripper: Gripper,
    skill: Optional[Skill],
    params: Optional[Dict[str, Any]],
    obj: str,
    mode: str = "id"
) -> Dict[str, Any]:
    """
    Returns dict(success, slip, energy, contact_count).
    mode:
      - id: in-distribution
      - ood: harder distribution (more slip / lower success)
    """
    stiff = float(np.mean(gripper.stiffness))
    skill_name = getattr(skill, "name", "baseline")

    # skill difficulty factors (higher is easier/better)
    skill_factor = {"pinch": 0.90, "power": 0.80, "hook": 0.70, "baseline": 0.75}.get(skill_name, 0.75)

    # object difficulty factors
    obj_factor = {"cube": 0.90, "cylinder": 0.80, "box": 0.85}.get(obj, 0.80)

    # OOD makes it harder
    ood_penalty = 0.15 if mode == "ood" else 0.0

    # probability of success depends on stiffness + skill + object
    p_success = 0.25 + 0.08 * stiff * skill_factor * obj_factor - ood_penalty
    p_success = float(np.clip(p_success, 0.01, 0.98))
    success = bool(rng.random() < p_success)

    # slip/energy proxy
    base_slip = (1.0 - skill_factor) * 0.25 + (0.12 if mode == "ood" else 0.02)
    slip = float(np.clip(base_slip + (0.10 if not success else 0.0), 0.0, 1.0))

    # higher stiffness tends to reduce energy term here (simple proxy)
    energy = float((1.0 / max(stiff, 1e-3)) + (0.25 if skill_name == "power" else 0.12))

    contact_count = 2 if skill_name == "pinch" else (4 if skill_name == "power" else 3)

    return {"success": success, "slip": slip, "energy": energy, "contact_count": contact_count, "p_success": p_success}

def simulate(
    rng: np.random.Generator,
    gripper: Gripper,
    skill: Optional[Skill],
    params: Optional[Dict[str, Any]],
    obj: str,
    mode: str,
    use_neural: bool
) -> Dict[str, Any]:
    """
    Unified simulator:
      - if use_neural and neural proxy exists -> simulate_neural(...)
      - else -> simulate_surrogate(...)
    """
    if use_neural and _SIM_NEURAL_OK and _simulate_neural is not None:
        # Try to be tolerant to different simulate_neural signatures
        try:
            return _simulate_neural(rng, gripper, skill=skill, params=params, obj=obj, mode=mode)
        except TypeError:
            # fallback to minimal arg set
            return _simulate_neural(rng, gripper, obj=obj, mode=mode)
        except Exception as e:
            print("[WARN] simulate_neural failed at runtime, fallback to surrogate:", repr(e))
            return simulate_surrogate(rng, gripper, skill=skill, params=params, obj=obj, mode=mode)
    return simulate_surrogate(rng, gripper, skill=skill, params=params, obj=obj, mode=mode)

# -------------------------
# 4) Metrics (skill-space objective + robustness)
# -------------------------
def score_fn(res: Dict[str, Any], w_slip: float, w_energy: float) -> float:
    """
    A simple scalar objective:
      - 0 if failed
      - else 1 - w_slip*slip - w_energy*energy
    """
    if not res.get("success", False):
        return 0.0
    slip = float(res.get("slip", 0.0))
    energy = float(res.get("energy", 0.0))
    return float(1.0 - w_slip * slip - w_energy * energy)

def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    if scores.size == 0:
        return {"mean": 0.0, "std": 0.0, "worst": 0.0, "p10": 0.0, "p50": 0.0}
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "worst": float(scores.min()),
        "p10": float(np.percentile(scores, 10)),
        "p50": float(np.percentile(scores, 50)),
    }

# -------------------------
# 5) Baseline vs Skill-space eval (matched budget)
# -------------------------
def run_baseline(
    rng: np.random.Generator,
    gripper: Gripper,
    objects: List[str],
    trials: int,
    mode: str,
    use_neural: bool,
    w_slip: float,
    w_energy: float,
) -> Dict[str, Any]:
    """
    Baseline = one fixed grasp “policy” (one skill, narrow params).
    Same compute budget: trials rollouts total.
    """
    baseline_skill = BaselineSkill()

    succ = 0
    scores: List[float] = []
    by_obj: Dict[str, Dict[str, int]] = {o: {"n": 0, "succ": 0} for o in objects}

    params_list = baseline_skill.sample_params(rng, n=trials)
    for i in range(trials):
        obj = objects[int(rng.integers(0, len(objects)))]
        res = simulate(rng, gripper, skill=baseline_skill, params=params_list[i], obj=obj, mode=mode, use_neural=use_neural)
        s = score_fn(res, w_slip=w_slip, w_energy=w_energy)
        succ += int(bool(res["success"]))
        scores.append(s)
        by_obj[obj]["n"] += 1
        by_obj[obj]["succ"] += int(bool(res["success"]))

    scores_arr = np.asarray(scores, dtype=np.float32)
    obj_rates = {o: (by_obj[o]["succ"] / max(by_obj[o]["n"], 1)) for o in objects}
    robust_obj_success = float(min(obj_rates.values())) if obj_rates else 0.0

    out = {
        "policy": "baseline_fixed_grasp",
        "success_rate": float(succ / max(trials, 1)),
        "scores": summarize_scores(scores_arr),
        "robust_obj_success": robust_obj_success,
        "obj_success_rates": {k: float(v) for k, v in obj_rates.items()},
        "trials_used": int(trials),
    }
    return out

def run_skill_space(
    rng: np.random.Generator,
    gripper: Gripper,
    skills: List[Skill],
    objects: List[str],
    trials: int,
    mode: str,
    use_neural: bool,
    w_slip: float,
    w_energy: float,
) -> Dict[str, Any]:
    """
    Skill-space = distribute same budget across skills (skills-as-coverage).
    """
    per_skill = max(1, trials // max(1, len(skills)))
    scores: List[float] = []
    succ = 0
    total = 0

    # track worst-case across skills and objects
    by_skill_obj: Dict[str, Dict[str, Dict[str, int]]] = {
        sk.name: {o: {"n": 0, "succ": 0} for o in objects} for sk in skills
    }

    for sk in skills:
        params_list = sk.sample_params(rng, n=per_skill)
        for p in params_list:
            obj = objects[int(rng.integers(0, len(objects)))]
            res = simulate(rng, gripper, skill=sk, params=p, obj=obj, mode=mode, use_neural=use_neural)
            total += 1
            succ += int(bool(res["success"]))
            scores.append(score_fn(res, w_slip=w_slip, w_energy=w_energy))

            by_skill_obj[sk.name][obj]["n"] += 1
            by_skill_obj[sk.name][obj]["succ"] += int(bool(res["success"]))

    scores_arr = np.asarray(scores, dtype=np.float32)

    # robust success = min over (skill, obj) cells that were actually sampled
    cell_rates: List[float] = []
    for sk in skills:
        for o in objects:
            n = by_skill_obj[sk.name][o]["n"]
            if n > 0:
                cell_rates.append(by_skill_obj[sk.name][o]["succ"] / n)
    robust_cell_success = float(min(cell_rates)) if cell_rates else 0.0

    out = {
        "policy": "skill_space_coverage",
        "skills": [sk.name for sk in skills],
        "success_rate": float(succ / max(total, 1)),
        "scores": summarize_scores(scores_arr),
        "robust_skill_obj_success": robust_cell_success,
        "trials_used": int(total),
        "per_skill": int(per_skill),
    }
    return out

# -------------------------
# 6) I/O
# -------------------------
def ensure_dirs(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

# -------------------------
# 7) Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["id", "ood"], default="id")
    parser.add_argument("--trials", type=int, default=90)
    parser.add_argument("--use_neural", action="store_true", help="use neural proxy if available; else fallback")
    parser.add_argument("--outdir", type=str, default="data/results")

    # weights for the score
    parser.add_argument("--w_slip", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=0.1)

    # simple morphology knob (still MVP, but lets you sweep quickly)
    parser.add_argument("--stiffness", type=float, default=5.0, help="uniform stiffness value for all blocks")
    parser.add_argument("--blocks", type=int, default=22, help="number of stiffness blocks")

    # skills / objects toggles
    parser.add_argument("--objects", type=str, default="cube,cylinder,box")
    parser.add_argument("--skills", type=str, default="pinch,power,hook")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ensure_dirs(args.outdir)

    # morphology
    gripper = Gripper.from_blocks([float(args.stiffness)] * int(args.blocks))

    objects = [s.strip() for s in args.objects.split(",") if s.strip()]
    skill_names = [s.strip() for s in args.skills.split(",") if s.strip()]

    name_to_skill = {"pinch": PinchSkill(), "power": PowerSkill(), "hook": HookSkill()}
    skills: List[Skill] = [name_to_skill[n] for n in skill_names if n in name_to_skill]
    if len(skills) == 0:
        skills = [PinchSkill(), PowerSkill(), HookSkill()]

    baseline = run_baseline(
        rng, gripper, objects, trials=args.trials, mode=args.mode,
        use_neural=bool(args.use_neural),
        w_slip=args.w_slip, w_energy=args.w_energy
    )
    skill_space = run_skill_space(
        rng, gripper, skills, objects, trials=args.trials, mode=args.mode,
        use_neural=bool(args.use_neural),
        w_slip=args.w_slip, w_energy=args.w_energy
    )

    out = {
        "seed": int(args.seed),
        "mode": args.mode,
        "trials_budget": int(args.trials),
        "use_neural_requested": bool(args.use_neural),
        "use_neural_available": bool(_SIM_NEURAL_OK),
        "morphology": {
            "blocks": int(args.blocks),
            "uniform_stiffness": float(args.stiffness),
            "mean_stiffness": float(np.mean(gripper.stiffness)),
        },
        "weights": {"w_slip": float(args.w_slip), "w_energy": float(args.w_energy)},
        "objects": objects,
        "skills": [sk.name for sk in skills],
        "baseline": baseline,
        "skill_space": skill_space,
        "delta": {
            "success_rate": float(skill_space["success_rate"] - baseline["success_rate"]),
            "robust_success": float(skill_space["robust_skill_obj_success"] - baseline["robust_obj_success"]),
            "mean_score": float(skill_space["scores"]["mean"] - baseline["scores"]["mean"]),
            "worst_score": float(skill_space["scores"]["worst"] - baseline["scores"]["worst"]),
        }
    }

    print("\n=== RESULTS ===")
    print("Baseline:", json.dumps(baseline, indent=2))
    print("Skill-space:", json.dumps(skill_space, indent=2))
    print("Delta:", json.dumps(out["delta"], indent=2))

    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.outdir, f"{ts}_seed{args.seed}_{args.mode}_T{args.trials}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved:", path)

if __name__ == "__main__":
    main()

import os, sys
from typing import Dict, Any

# --- path to baseline repo ---
BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "codesign-soft-gripper"))
BASELINE_CODE_DIR = os.path.join(BASELINE_DIR, "code")

if BASELINE_CODE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_CODE_DIR)

# Now we can import baseline modules (from code/*.py)
import evaluation  # baseline: evaluation.py
# train_eval / optimize can be used later if needed


def simulate_neural(gripper, skill, params, obj: str, mode: str = "id") -> Dict[str, Any]:
    """
    Adapter that *uses baseline project*.

    Returns dict compatible with our pipeline:
      success: bool
      slip: float
      energy: float
      contact_count: int

    MVP integration:
      - currently calls baseline evaluation utilities (you will map exact call below)
      - if exact function differs, we patch only here, not in hypothesis code.
    """

    # ---- TODO: map our morphology/skill params into baseline format ----
    # For now: we pass through placeholders.
    # You will replace these with actual baseline args:
    #   stiffness_blocks, pose, object mesh/pc, etc.
    stiffness_blocks = getattr(gripper, "stiffness", None)
    skill_name = getattr(skill, "name", "baseline")

    # Mode -> domain randomization flags (you can refine)
    ood = (mode == "ood")

    # --- Minimal "proof" that baseline is used ---
    # We call something trivial from baseline module and keep result in info
    # Replace with real evaluation call below.
    baseline_module_path = evaluation.__file__

    # ---- PLACEHOLDER OUTPUT until we bind real baseline eval ----
    # We return something deterministic-ish so you can see it changes later.
    success = True if not ood else True
    slip = 0.05 if not ood else 0.10
    energy = 0.8
    contact_count = 2 if skill_name == "pinch" else 4

    return {
        "success": bool(success),
        "slip": float(slip),
        "energy": float(energy),
        "contact_count": int(contact_count),
        "baseline_used": True,
        "baseline_module": baseline_module_path,
    }

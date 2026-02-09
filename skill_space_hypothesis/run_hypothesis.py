print(">>> run_hypothesis.py started")

from skills.pinch import PinchSkill
from skills.power import PowerSkill
from skills.hook import HookSkill

from morphology.gripper import Gripper
from physics.surrogate import simulate

from experiments.skill_space_eval import evaluate_skill_space
from experiments.baseline_object_task import run_baseline


def main():
    print(">>> main() entered")

    # 1. морфология
    gripper = Gripper(stiffness_blocks=[5.0] * 22)

    # 2. объекты (пока заглушки)
    objects = ["cube", "cylinder", "box"]

    # 3. навыки
    skills = [PinchSkill(), PowerSkill(), HookSkill()]

    # 4. baseline
    baseline = run_baseline(
        gripper=gripper,
        objects=objects,
        simulator=simulate
    )

    print("Baseline:", baseline)

    # 5. skill-space
    skill_space = evaluate_skill_space(
        gripper=gripper,
        skills=skills,
        objects=objects,
        simulator=simulate,
        n_params=16
    )

    print("Skill-space:", skill_space)


if __name__ == "__main__":
    main()

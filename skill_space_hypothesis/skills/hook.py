from skillspace.pinch import PinchSkill
from skillspace.power import PowerSkill
from skillspace.hook import HookSkill

import numpy as np

class HookSkill(Skill):
    name = "hook"

    def sample_params(self, n=16):
        return [
            {
                "force": np.random.uniform(2.0, 6.0),
                "angle": np.random.uniform(-0.5, 0.5),
            }
            for _ in range(n)
        ]

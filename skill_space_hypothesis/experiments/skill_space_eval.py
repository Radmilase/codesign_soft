def evaluate_skill_space(gripper, skills, objects, simulator):
    scores = []
    for skill in skills:
        params = skill.sample_params(32)
        for p in params:
            for obj in objects:
                res = rollout(gripper, skill, p, obj, simulator)
                scores.append(skill_score(res))
    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "worst": np.min(scores)
    }

def speedrun_reward(info, prev_info, default_reward):
    """
    Speedrun mode: maximize progress, minimize time spent.

    - Reward forward movement (x_pos increasing)
    - Big bonus for reaching the flag, scaled by time remaining
      (more time left = faster completion = bigger bonus)
    - Small penalty every step (encourages haste)
    - Penalty for dying
    """
    reward = 0

    # Progress reward: how much further right since last step
    progress = info["x_pos"] - prev_info["x_pos"]
    reward += progress

    # Time pressure: lose a tiny bit every step, encouraging speed
    reward -= 0.1

    # Big bonus for finishing, weighted by time remaining
    if info["flag_get"]:
        reward += 100 + info["time"] * 2  # more time left = bigger bonus

    # Penalty for dying
    if info["life"] < prev_info["life"]:
        reward -= 50

    return reward


def completionist_reward(info, prev_info, default_reward):
    """
    Completionist mode: maximize score, coins, and exploration.

    - Reward forward movement (still need to progress through the level)
    - Reward coin collection
    - Reward score increases (covers enemy stomps, item pickups, etc.)
    - Bonus for reaching the flag (but NOT scaled by time - speed doesn't matter)
    - No per-step time penalty - taking your time is fine
    - Penalty for dying
    """
    reward = 0

    # Still need forward progress to advance through the level
    progress = info["x_pos"] - prev_info["x_pos"]
    reward += progress * 0.5  # weighted less than speedrun - not the main goal

    # Reward picking up coins
    coins_gained = info["coins"] - prev_info["coins"]
    reward += coins_gained * 10

    # Reward score increases (enemies defeated, items collected, etc.)
    score_gained = info["score"] - prev_info["score"]
    reward += score_gained * 0.1

    # Flag bonus - flat, no time scaling
    if info["flag_get"]:
        reward += 100

    # Penalty for dying
    if info["life"] < prev_info["life"]:
        reward -= 50

    return reward
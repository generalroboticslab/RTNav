import re
import sys


def get_stats(log_path, max_episodes=None):
    with open(log_path, "r") as f:
        lines = f.readlines()

    step_pattern = re.compile(r"Step:\s*(\d+)\s*\|\s*Mode:\s*\w+\s*\|\s*Action:\s*(\d+)")
    success_pattern = re.compile(r"Success rate:\s*[\d.]+%\s*\((\d+)\s*out of\s*(\d+)\)")

    total_episodes = 0
    successes = 0
    timeouts = 0
    false_positives = 0

    prev_success_count = None
    current_max_step = -1
    called_action_0 = False
    waiting_for_success_rate = False

    for line in lines:
        step_match = step_pattern.search(line)
        success_match = success_pattern.search(line)

        if step_match:
            step_num = int(step_match.group(1))
            action = int(step_match.group(2))

            # New episode detected (step resets to 0)
            if step_num == 0 and current_max_step >= 0:
                # Previous episode ended without action 0 and no success rate line
                if not called_action_0:
                    if current_max_step >= 499:
                        timeouts += 1
                    else:
                        # Edge case: episode ended abruptly (treat as timeout if step >= 499)
                        timeouts += 1
                    total_episodes += 1

                if max_episodes is not None and total_episodes >= max_episodes:
                    break

                # Reset for new episode
                current_max_step = 0
                called_action_0 = False
                waiting_for_success_rate = False
            else:
                current_max_step = max(current_max_step, step_num)

            if action == 0:
                called_action_0 = True
                waiting_for_success_rate = True

        if success_match and waiting_for_success_rate:
            current_success_count = int(success_match.group(1))
            total_eps = int(success_match.group(2))

            total_episodes += 1
            if prev_success_count is None or current_success_count > prev_success_count:
                successes += 1
            else:
                false_positives += 1

            prev_success_count = current_success_count
            waiting_for_success_rate = False
            called_action_0 = False
            current_max_step = -1  # will be reset on next Step: 0

            if max_episodes is not None and total_episodes >= max_episodes:
                break

    # Handle last episode if it didn't end with a success rate line
    if (max_episodes is None or total_episodes < max_episodes) and current_max_step >= 0 and not called_action_0:
        total_episodes += 1
        if current_max_step >= 499:
            timeouts += 1
        else:
            timeouts += 1  # incomplete / timed out

    if total_episodes == 0:
        print("No episodes found.")
        return

    print(f"Total episodes: {total_episodes}")
    print(f"  Successes:       {successes} ({100 * successes / total_episodes:.2f}%)")
    print(f"  Timeouts:        {timeouts} ({100 * timeouts / total_episodes:.2f}%)")
    print(f"  False positives: {false_positives} ({100 * false_positives / total_episodes:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stats.py logs/406_v0.txt [N]")
        sys.exit(1)
    n = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    get_stats(sys.argv[1], max_episodes=n)


# detection threshold 0.3 (incomplete)
# Total episodes: 1798
#   Successes:       530 (29.48%)
#   Timeouts:        301 (16.74%)
#   False positives: 967 (53.78%)

# detection threshold 0.4:
# Total episodes: 3000
#   Successes:       972 (32.40%)
#   Timeouts:        663 (22.10%)
#   False positives: 1365 (45.50%)
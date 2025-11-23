import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# load scalar from TensorBoard event files
def load_tb_scalar(log_dir, tag):
    # Search inside subfolders
    event_files = glob.glob(os.path.join(log_dir, "**", "events*"), recursive=True)

    if not event_files:
        print(f"[WARN] No tensorboard events in {log_dir}")
        return [], []

    event_dir = os.path.dirname(event_files[0])

    ea = EventAccumulator(event_dir)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        print(f"[WARN] Tag '{tag}' not found in {log_dir}")
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# Best experiment directories
BEST_DQN = "logs_DQN-6"
BEST_PPO = "ppo_logs_PPO-8"
BEST_A2C = "a2c_logs_A2C-9"
BEST_REINFORCE = "reinforce_logs"


# Cumulative Reward Comparison
def plot_cumulative_rewards():
    plt.figure(figsize=(12, 6))

    for label, log_dir in [
        ("DQN-6", BEST_DQN),
        ("PPO-8", BEST_PPO),
        ("A2C-9", BEST_A2C),
    ]:
        steps, vals = load_tb_scalar(log_dir, "rollout/ep_rew_mean")
        if steps:
            plt.plot(steps, vals, label=label)

    # REINFORCE stored manually
    r_path = os.path.join(BEST_REINFORCE, "reinforce_rewards.txt")
    if os.path.exists(r_path):
        with open(r_path) as f:
            r = [float(x.strip()) for x in f.readlines()]
        plt.plot(range(len(r)), r, label="REINFORCE-7")

    plt.title("Cumulative Reward Comparison (Best Models)")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot_cumulative_rewards.png", dpi=300)
    print("Saved: plot_cumulative_rewards.png")
    plt.show()


# PPO Entropy Curve
def plot_ppo_entropy():
    steps, vals = load_tb_scalar(BEST_PPO, "train/entropy_loss")
    if not steps:
        print("⚠ PPO entropy not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(steps, vals)
    plt.title("PPO Policy Entropy")
    plt.xlabel("Training Steps")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.savefig("plot_ppo_entropy.png", dpi=300)
    print("Saved: plot_ppo_entropy.png")
    plt.show()


# DQN Loss Curve
def plot_dqn_loss():
    steps, vals = load_tb_scalar(BEST_DQN, "train/loss")
    if not steps:
        print("⚠ DQN loss not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(steps, vals)
    plt.title("DQN Loss Curve")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plot_dqn_loss.png", dpi=300)
    print("Saved: plot_dqn_loss.png")
    plt.show()


# A2C Reward Curve
def plot_a2c_rewards():
    steps, vals = load_tb_scalar(BEST_A2C, "rollout/ep_rew_mean")

    if not steps:
        print("⚠ A2C reward not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(steps, vals)
    plt.title("A2C Reward Progression")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("plot_a2c_rewards.png", dpi=300)
    print("Saved: plot_a2c_rewards.png")
    plt.show()


# REINFORCE Loss Curve
def plot_reinforce_loss():
    path = os.path.join(BEST_REINFORCE, "reinforce_losses.txt")

    if not os.path.exists(path):
        print("⚠ No REINFORCE loss file found.")
        return

    with open(path) as f:
        losses = [float(x.strip()) for x in f.readlines()]

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("REINFORCE Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plot_reinforce_loss.png", dpi=300)
    print("Saved: plot_reinforce_loss.png")
    plt.show()

if __name__ == "__main__":
    print("\nGenerating all comparison plots...\n")
    plot_cumulative_rewards()
    plot_ppo_entropy()
    plot_dqn_loss()
    plot_a2c_rewards()
    plot_reinforce_loss()
    print("\nAll plots successfully saved!\n")

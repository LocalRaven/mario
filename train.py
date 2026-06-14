import os
import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from preprocess import preprocess_frame
from model import MarioNet
from agent import FrameStack, Agent, ReplayMemory
from rewards import speedrun_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Setup: lock to World 1-1 ---
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = MarioNet(input_channels=4, num_actions=env.action_space.n).to(device)
agent = Agent(model, num_actions=env.action_space.n)
memory = ReplayMemory(capacity=100_000)
stack = FrameStack(num_frames=4)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_EPISODES = 50_000
SAVE_EVERY = 100
STUCK_LIMIT = 100  # end episode if no progress for this many steps

for episode in range(NUM_EPISODES):

    state, info = env.reset()
    frame = preprocess_frame(state)
    stacked_state = stack.reset(frame)

    prev_info = dict(info)
    prev_info["max_x_pos"] = info["x_pos"]
    stuck_counter = 0

    total_reward = 0
    total_loss = 0
    loss_count = 0

    while True:
        action = agent.choose_action(stacked_state)

        next_state, default_reward, done, truncated, info = env.step(action)
        next_frame = preprocess_frame(next_state)
        stacked_next_state = stack.push(next_frame)

        reward = speedrun_reward(info, prev_info, default_reward)

        # --- Stuck detection ---
        if info["x_pos"] <= prev_info["max_x_pos"]:
            stuck_counter += 1
        else:
            stuck_counter = 0
        prev_info["max_x_pos"] = max(info["x_pos"], prev_info["max_x_pos"])

        if stuck_counter > STUCK_LIMIT:
            done = True

        memory.push(stacked_state, action, reward, stacked_next_state, done)

        loss = agent.learn(memory, batch_size=64)
        if loss is not None:
            total_loss += loss
            loss_count += 1

        total_reward += reward
        stacked_state = stacked_next_state
        prev_info = info
        prev_info["max_x_pos"] = max(info["x_pos"], prev_info.get("max_x_pos", 0))

        if done:
            break

    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    print(f"Episode {episode + 1}/{NUM_EPISODES} | "
          f"Reward: {total_reward:.2f} | "
          f"x_pos: {info['x_pos']} | "
          f"Epsilon: {agent.epsilon:.4f} | "
          f"Avg Loss: {avg_loss:.4f}")

    if (episode + 1) % SAVE_EVERY == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"mario_speedrun_{episode + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  -> Saved checkpoint: {checkpoint_path}")

env.close()
import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from preprocess import preprocess_frame
from model import MarioNet
from agent import FrameStack

CHECKPOINT_PATH = "checkpoints/mario_speedrun_400.pth"  # change to whichever checkpoint you want to show

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = MarioNet(input_channels=4, num_actions=env.action_space.n)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()  # tell the network "no more learning, just predict"

stack = FrameStack(num_frames=4)

state, info = env.reset()
frame = preprocess_frame(state)
stacked_state = stack.reset(frame)

done = False
while not done:
    with torch.no_grad():
        q_values = model(stacked_state.unsqueeze(0))
        action = torch.argmax(q_values).item()  # always exploit - no randomness

    state, reward, done, truncated, info = env.step(action)
    frame = preprocess_frame(state)
    stacked_state = stack.push(frame)

    env.render()

print("Final x_pos:", info["x_pos"])
print("Flag reached:", info["flag_get"])
env.close()
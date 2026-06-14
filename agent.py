from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class FrameStack:
    """
    Keeps the last `num_frames` preprocessed frames stacked together,
    so the network can perceive motion.
    """

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        """Fill the stack with copies of the first frame."""
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self._get_stack()

    def push(self, frame):
        """Add a new frame, automatically dropping the oldest."""
        self.frames.append(frame)
        return self._get_stack()

    def _get_stack(self):
        # Stack along a new dimension: result shape = (num_frames, 84, 84)
        return torch.cat(list(self.frames), dim=0)
    

class Agent:
    def __init__(self, model, num_actions, epsilon=1.0, epsilon_min=0.1,
                 epsilon_decay=0.99999, gamma=0.9, lr=0.00025):
        self.model = model
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma  # how much we care about future rewards vs immediate ones

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                device = next(self.model.parameters()).device
                state_batch = state.unsqueeze(0).to(device)
                q_values = self.model(state_batch)
                action = torch.argmax(q_values).item()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def learn(self, memory, batch_size=32):
        if len(memory) < batch_size:
            return None

        device = next(self.model.parameters()).device
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
           next_q = self.model(next_states).max(1)[0]
           target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
class ReplayMemory:
    """
    Stores (state, action, reward, next_state, done) tuples and lets us
    sample random batches of them for training.
    """

    def __init__(self, capacity=100_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        # Unzip the batch into separate tensors
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
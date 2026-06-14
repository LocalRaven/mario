import numpy as np
import torch
from torchvision import transforms as T

def preprocess_frame(frame):
    """Convert a raw RGB game frame into a small grayscale tensor."""
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((84, 84)),
        T.ToTensor()
    ])
    return transform(frame)
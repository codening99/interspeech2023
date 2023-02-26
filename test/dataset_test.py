import torch
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
import torchaudio

spec_score = [0.5, 0.7, 0.2]
spec_score = spec_score > 0.5
print(spec_score)




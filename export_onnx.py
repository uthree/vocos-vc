import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
from module.convertor import VoiceConvertor

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outputs', default="./onnx/")
parser.add_argument('-p', '--model-path', default="./model.pt")

args = parser.parse_args()

device = torch.device('cpu')

model = VoiceConvertor()
model.load_state_dict(torch.load(args.model_path, map_location=device))

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

print("Exporting Onnx...")

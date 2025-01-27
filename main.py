import torch
from webcame import GlossesRecognition
import os 
from torch import nn
from data.model import Model
import json

def main():
    
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
  else:
    device = torch.device("cpu")
    print("CPU")
  model = Model().to(device)
  model.load_state_dict(torch.load(os.path.join("data","model_parameters.pth"), map_location=device))
  with open(os.path.join("data","data.json"), "r") as f:
      index_to_label = json.load(f)
      
  index_to_label=index_to_label['index-to-label']

  
  GlossesRecognition(
    model=model,
    dictanory=index_to_label,
    num_frames=15,
    device=device,
    maximum_glosses_length=3,
    length_of_frames_predictions=30
    ).start_recognition()
  
if __name__ == "__main__":
    main()
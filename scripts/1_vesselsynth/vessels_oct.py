import torch
from veritas.synth import VesselSynth
torch.no_grad()

if __name__ == "__main__":
    VesselSynth(experiment_number=1).synth()
import iseeu

import torch

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")
    
    net = iseeu.ISeeU(iseeu.ISeeUModule, [2, 4, 6, 3]).to(device)

    # TODO:
    # Implement inferencing
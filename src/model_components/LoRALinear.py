import torch

class LoRALinear(torch.nn.Module):
    def __init__(self, in_features, out_features, r, device=None, dtype=None):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, r, bias=False, device=device, dtype=dtype) # self.linear1.weight r, in_features
        self.linear2 = torch.nn.Linear(r, out_features, device=device, dtype=dtype) # self.linear2.weight out_features, r 
    
    def forward(self, x):
        return self.linear2(self.linear1(x))
    
    def weight(self):
        return self.linear2.weight @ self.linear1.weight
    
    def bias(self):
        return self.linear2.bias
                
    
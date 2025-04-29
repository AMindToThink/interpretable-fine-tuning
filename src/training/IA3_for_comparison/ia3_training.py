#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Use only GPU 0
import torch
print(torch.cuda.device_count())  # Should print 1
#%%
import torch

# check if CUDA is available
torch.zeros(1).cuda()
train_on_gpu = torch.cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')
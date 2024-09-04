import torch

lr = 1e-5

path = './inter-est_pooled'
save_path = '/home/ryn_mote/Misc/vlm_with_pooled_for_text_genrec/inter-est'

epochs = 1
batch_size = 8

device = 'cuda'
dtype = torch.bfloat16

max_tokens = 32


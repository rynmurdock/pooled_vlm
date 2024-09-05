import torch

lr = 1e-5

path = '/home/ryn_mote/Misc/vlm_with_pooled_for_text_genrec/inter-est_CLIP_aesth_7000'
save_path = '/home/ryn_mote/Misc/vlm_with_pooled_for_text_genrec/inter-est_CLIP_aesth'

epochs = 1
batch_size = 16

device = 'cuda'
dtype = torch.bfloat16

max_tokens = 32


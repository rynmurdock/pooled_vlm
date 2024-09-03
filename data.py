from datasets import load_dataset
import torch
import torchvision
import config

def my_collate(batch):
    try:
      img = [item['image'] for item in batch]
      text = ['<image>'+item['conversations'][1]['value'] for item in batch]
    except:
      return None
    return {'image':img, 'text':text}

ds = load_dataset("lmms-lab/LLaVA-ReCap-118K", split='train', streaming=True).shuffle(seed=7, buffer_size=2)
dataloader = torch.utils.data.DataLoader(ds, num_workers=2, collate_fn=my_collate, batch_size=config.batch_size)


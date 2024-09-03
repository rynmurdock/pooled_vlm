from datasets import load_dataset
import torch
import torchvision

def my_collate(batch):
    try:
      img = [item['image'] for item in batch]
      text = [item['conversations'][1]['value'] for item in batch]
    except:
      return None
    return {'image':img, 'text':text}

ds = load_dataset("lmms-lab/LLaVA-ReCap-118K", split='train', streaming=True).shuffle(seed=42, buffer_size=2)
dataloader = torch.utils.data.DataLoader(ds, num_workers=2, collate_fn=my_collate)


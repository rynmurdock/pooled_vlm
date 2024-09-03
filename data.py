from datasets import load_dataset
import torch
import config
from model import load_image

def my_collate(batch):
    try:
      img = [item['image'] for item in batch]
      img = torch.cat([load_image(pil_image=i, image_file=None) for i in img])
      text = ['''<|user|><img><IMG_CONTEXT></img><|end|><|assistant|>'''+item['conversations'][1]['value'] for item in batch]
    except:
      return None
    return {'image':img, 'text':text}

ds = load_dataset("lmms-lab/LLaVA-ReCap-118K", split='train', streaming=True).shuffle(seed=7, buffer_size=4)
dataloader = torch.utils.data.DataLoader(ds, num_workers=32, collate_fn=my_collate, batch_size=config.batch_size)

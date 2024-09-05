from datasets import load_dataset
import torch
import config
from model import load_image

from diffusers.utils.loading_utils import load_image as dl_to_pil

def my_collate(batch):
    try:
      img = [item['image'] for item in batch]
      img = torch.cat([load_image(pil_image=i, image_file=None) for i in img])
      text = ['''<|user|><img><IMG_CONTEXT></img><|end|><|assistant|>'''+item['prompt'] for item in batch]
    except Exception as e:
      print(e)
      return None
    return {'image':img, 'text':text}

ds = load_dataset("stylebreeder/stylebreeder", split='2M_sample', streaming=True).shuffle(seed=7, buffer_size=1)
dataloader = torch.utils.data.DataLoader(ds, num_workers=32, collate_fn=my_collate, batch_size=config.batch_size)

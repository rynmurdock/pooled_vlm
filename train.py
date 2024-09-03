

########################################
# python -m train
###########################################
# see https://huggingface.co/OpenGVLab/InternVL2-4B/




import torch

from data import dataloader
from model import model, tokenizer, optimizer, load_image
import config

def get_loss(model, input):
    ids = tokenizer(input['text'], return_tensors='pt', padding=True, truncation=True, max_length=128).to(config.device)
    pixel_values = torch.cat([load_image(pil_image=i, image_file=None) for i in input['image']]).to(config.device, config.dtype)
    pixel_values = torch.nn.functional.interpolate(pixel_values, (224, 224))
    print(pixel_values.shape, 'pixel_values shape')
    with torch.cuda.amp.autocast(enabled=True, dtype=config.dtype):
        output = model(**ids, labels=ids.input_ids, pixel_values=pixel_values)
    
    return output.loss


for epoch in range(config.epochs):
    for sample in iter(dataloader):
        print(sample)
        if sample is None:
            continue
        loss = get_loss(model, sample)
        print(loss.item()) 
        # TODO generation validation

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        break






import torch

from data import dataloader
from model import model, tokenizer, optimizer, load_image
import config

def get_loss(model, input):
    ids = tokenizer(input['text'], return_tensors='pt', )
    pixel_values = torch.cat([load_image(pil_image=i, image_file=None) for i in input['image']])
    assert 'labels' in ids
    output = model(**ids, pixel_values=pixel_values)
    
    return output.loss


for epoch in range(config.epochs):
    for sample in iter(dataloader):
        print(sample)
        if sample is None:
            continue
        loss = get_loss(model, sample)
        print(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        break






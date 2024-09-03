

########################################
# python -m train
###########################################
# see https://huggingface.co/OpenGVLab/InternVL2-4B/




import torch
from tqdm import tqdm

from data import dataloader
from model import model, tokenizer, optimizer, load_image
import config

def get_loss(model, input):
    ids = tokenizer(input['text'], return_tensors='pt', padding=True, truncation=True, max_length=config.max_tokens).to(config.device)
    # print(tokenizer.decode(ids.input_ids[0]))
    pixel_values = input['image'].to(config.device, config.dtype)
    pixel_values = torch.nn.functional.interpolate(pixel_values, (224, 224))
    with torch.cuda.amp.autocast(enabled=True, dtype=config.dtype):
        output = model(**ids, labels=ids.input_ids, pixel_values=pixel_values)
    
    return output.loss

scaler = torch.cuda.amp.GradScaler()

for epoch in range(config.epochs):
    for ind, sample in tqdm(enumerate(iter(dataloader))):
        if sample is None:
            continue

        if ind % 100 == 0:
            with torch.cuda.amp.autocast(enabled=True, dtype=config.dtype):
                response = model.chat(tokenizer=tokenizer, 
                                  pixel_values=torch.nn.functional.interpolate(
                                      load_image('/home/ryn_mote/Downloads/horse_style.png').to(config.device, config.dtype), 
                                      (224, 224)), 
                                  question='<image>\n ', 
                                  generation_config = dict(max_new_tokens=config.max_tokens, do_sample=True))
                print(response)

                response = model.chat(tokenizer=tokenizer, 
                                  pixel_values=torch.nn.functional.interpolate(
                                      load_image('/home/ryn_mote/Downloads/1200px-Andrzej_Person_Kancelaria_Senatu.jpg').to(config.device, config.dtype), 
                                      (224, 224)), 
                                  question='<image>\n ', 
                                  generation_config = dict(max_new_tokens=config.max_tokens, do_sample=True))
                print(response)

        loss = get_loss(model, sample)
        print(loss.item())

        scaler.scale(loss).backward()

        optimizer.step()
        optimizer.zero_grad()


        if ind % 1000 == 0:
            model.save_pretrained(config.save_path, from_pt=True) 




# TODO chart loss


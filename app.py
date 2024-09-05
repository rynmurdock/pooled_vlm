import torch
from PIL import Image

import random
import pandas as pd
import gradio as gr
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
import time
import torch
from matplotlib import pyplot as plt

from model import model, tokenizer, load_image

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = 'cuda'
dtype = torch.bfloat16

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=dtype, variant="fp16").to(device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")





with torch.cuda.amp.autocast(True, dtype):
    # extract eos/mean embedding
    pixel_values = load_image(image_file='blank.png', max_num=1).to(device)
    base_embed = model.extract_feature(pixel_values.to(dtype)).detach().float()



def get_text(embed):
    with torch.cuda.amp.autocast(True, dtype):
        generation_config = dict(max_new_tokens=32, do_sample=True, 
                                                    temperature=.5, top_p=.92)

        # single-image single-round conversation (单图单轮对话)
        pixel_values = 0
        question = '''''' # not really used # TODO & pixel_values as well
        response = model.chat(tokenizer, pixel_values, question, generation_config, visual_features=embed.to(dtype))
        print(response)
        return response

def get_image(text):
    return pipe(text, num_inference_steps=8, guidance_scale=0).images[0]

def get_embed(img):
    with torch.cuda.amp.autocast(True, dtype):
        # extract eos/mean embedding
        pixel_values = load_image(image_file='', pil_image=img, max_num=1).to(device)
        embed = model.extract_feature(pixel_values.to(dtype))
    return embed.float()



prompt_list = [p for p in list(set(
                pd.read_csv('/home/ryn_mote/Misc/twitter_prompts.csv').iloc[:, 1].tolist())) if type(p) == str]
random.shuffle(prompt_list)



NOT_calibrate_prompts = [
    'an abstract painting',
    'unique streetwear  design that blends the old with the new. Combine bold, urban typography with retro graphics, taking inspiration from distressed signage and graffiti. Use a range of earthy tones to give the design a vintage aesthetic, while adding a modern twist with a stylistic rendering of the graphics',
    'a photo of hell',
    ''
    ]

calibrate_prompts = [
    "4k photo",
    'surrealist art',
    'a psychedelic, fractal view',
    'a beautiful collage',
    'an intricate portrait',
    'an impressionist painting',
    'abstract art',
    'an eldritch image',
    'a sketch',
    'a city full of darkness and graffiti',
    'a black & white photo',
    'a brilliant, timeless tarot card of the world',
    '''eternity: a timeless, vivid painted portrait by ryan murdock''',
    '''a simple, timeless, & dark charcoal on canvas: death itself by ryan murdock''',
    '''a painted image with gorgeous red gradients: Persephone by ryan murdock''',
    '''a simple, timeless, & dark photo with gorgeous gradients: last night of my life by ryan murdock''',
    '''the sunflower -- a dark, simple painted still life by ryan murdock''',
    '''silence in the macrocosm -- a dark, intricate painting by ryan murdock''',
    '''beauty here -- a photograph by ryan murdock''',
    '''a timeless, haunting portrait: the necrotic jester''',
    '''a simple, timeless, & dark art piece with gorgeous gradients: serenity''',
    '''an elegant image of nature with gorgeous swirling gradients''',
    '''simple, timeless digital art with gorgeous purple spirals''',
    '''timeless digital art with gorgeous gradients: eternal slumber''',
    '''a simple, timeless image with gorgeous gradients''',
    '''a simple, timeless painted image of nature with beautiful gradients''',
    'a timeless, dark digital art piece with gorgeous gradients: the hanged man',
    '',
]



global_idx = 0
embs = []
ys = []

start_time = time.time()

def next_image():
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            prompt = calibrate_prompts.pop(0)
            print(f'######### Calibrating with sample: {prompt} #########')

            image = get_image(prompt)


            ####### optional step; we could take the prior output instead
            with torch.cuda.amp.autocast():
                embed = get_embed(image)
            #######

            embs.append(embed)
            return image, prompt
        else:
            print('######### Roaming #########')

            # sample only as many negatives as there are positives
            indices = range(len(ys))                
            pos_indices = [i for i in indices if ys[i] > .5]
            neg_indices = [i for i in indices if ys[i] <= .5]
            
            mini = min(len(pos_indices), len(neg_indices))
            
            if mini < 1:
                feature_embs = torch.stack([torch.randn(1280), torch.randn(1280)])
                ys_t = [0, 1]
                print('Not enough ratings.')
            else:
                # indices = random.sample(pos_indices, mini) + random.sample(neg_indices, mini)
                ys_t = [ys[i] for i in indices]
                feature_embs = torch.stack([embs[e][0, 0].detach().cpu() for e in indices]).squeeze()

                # # balance pos/negatives?
                # for e in indices:
                #     nw = (len(indices) / len(neg_indices))
                #     w = (len(indices) / len(pos_indices))
                #     feature_embs[e] = feature_embs[e] * w if ys_t[e] > .5 else feature_embs[e] * nw
                
                # if len(pos_indices) > 8:
                #    to_drop = pos_indices.pop(0)
                #    ys.pop(to_drop)
                #    embs.pop(to_drop)
                #    print('\n\n\ndropping\n\n\n')
                # elif len(neg_indices) > 8:
                #    to_drop = neg_indices.pop(0)
                #    ys.pop(to_drop)
                #    embs.pop(to_drop)
                #    print('\n\n\ndropping\n\n\n')
                
                
                # scaler = preprocessing.StandardScaler().fit(feature_embs)
                # feature_embs = scaler.transform(feature_embs)
                # ys_t = ys
                
                print(np.array(feature_embs).shape, np.array(ys_t).shape)
            
            # sol = LogisticRegression().fit(np.array(feature_embs), np.array(torch.tensor(ys_t).unsqueeze(1).float() * 2 - 1)).coef_
            # sol = torch.linalg.lstsq(torch.tensor(ys_t).unsqueeze(1).float()*2-1, torch.tensor(feature_embs).float(),).solution
            # neg_sol = torch.linalg.lstsq((torch.tensor(ys_t).unsqueeze(1).float() - 1) * -1, torch.tensor(feature_embs).float()).solution
            # sol = torch.tensor(sol, dtype=dtype).to(device)


            pos_sol = torch.stack([feature_embs[i] for i in range(len(ys_t)) if ys_t[i] > .5]).mean(0, keepdim=True).to(device, dtype)
            neg_sol = torch.stack([feature_embs[i] for i in range(len(ys_t)) if ys_t[i] < .5]).mean(0, keepdim=True).to(device, dtype)
            
            # could j have a base vector of a black image
            latest_pos = (random.sample([feature_embs[i] for i in range(len(ys_t)) if ys_t[i] > .5], 1)[0]).to(device, dtype)

            dif = pos_sol - neg_sol
            sol = latest_pos + ((dif / dif.std()) * latest_pos.std())

            print(sol.shape)
            

            text = get_text(sol)
            image = get_image(text)
            embed = get_embed(image)

            embs.append(embed)

            plt.close()
            plt.hist(sol.detach().cpu().float().flatten())
            plt.savefig('sol.jpg')


            plt.close()
            plt.hist(embed.detach().cpu().float().flatten())
            plt.savefig('embed.jpg')
            
            # torch.save(sol, f'./{start_time}.pt')
            return image, text
            






def start(_):
    return [
            gr.Button(value='Like', interactive=True), 
            gr.Button(value='Neither', interactive=True), 
            gr.Button(value='Dislike', interactive=True),
            gr.Button(value='Start', interactive=False),
            *next_image()
            ]


def choose(choice):
    global global_idx
    global_idx += 1
    if choice == 'Like':
        choice = 1
    elif choice == 'Neither':
        _ = embs.pop(-1)
        return next_image()
    else:
        choice = 0
    ys.append(choice)
    return next_image()

css = "div#output-image {height: 512px !important; width: 512px !important; margin:auto;}"
with gr.Blocks(css=css) as demo:
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:32'>You will callibrate for several prompts and then roam.</ div>''')
    with gr.Row(elem_id='output-image'):
        img = gr.Image(interactive=False, elem_id='output-image',)
    with gr.Row(elem_id='output-txt'):
        txt = gr.Textbox(interactive=False, elem_id='output-txt',)
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='Dislike', interactive=False,)
        b2 = gr.Button(value='Neither', interactive=False,)
        b1 = gr.Button(value='Like', interactive=False,)
        b1.click(
        choose, 
        [b1],
        [img, txt]
        )
        b2.click(
        choose, 
        [b2],
        [img, txt]
        )
        b3.click(
        choose, 
        [b3],
        [img, txt]
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4],
                 [b1, b2, b3, b4, img, txt])

demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀



# TODO use CLIP text encoder pooled & keep frozen


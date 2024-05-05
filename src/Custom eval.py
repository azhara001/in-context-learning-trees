# taking care of virtual environment dependencies 

#!pip install -q condacolab
#import condacolab
#condacolab.install()

# Mounting Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/in-context-learning-trees/src

## %%shell
# eval "$(conda shell.bash hook)" # copy conda command to shell
# conda activate in-context-learning
# python --version
# # conda deactivate

# Custom eval_script 
from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"

df = read_run_dir(run_dir)
# print(df)

run_id = "pretrained"  # if you train more models, replace with the run_id from the table above
task = 'decision_tree'

run_path = os.path.join(run_dir, task, run_id)
print(run_path)

import models
from eval import *
#get_run_metrics(run_path)
step = -1
model, conf = get_model_from_run(run_path, step)
# model = model.cuda().eval()
all_models = [model]
skip_baselines = False
if not skip_baselines:
    all_models += models.get_relevant_baselines(conf.training.task)
evaluation_kwargs = build_evals(conf)
save_path = os.path.join(run_path, "metrics.json")

# now calling the compute_evals function
all_metrics = {}

for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
    metrics = {}
    for model in all_models:
        #metrics[model.name] = eval_model(model, **kwargs)
        task_name = kwargs['task_name']
        data_name = kwargs['data_name']
        n_dims = kwargs['n_dims']
        n_points = kwargs['n_points']
        prompting_strategy = kwargs['prompting_strategy']
        num_eval_examples=1280
        batch_size=64
        data_sampler_kwargs={}
        task_sampler_kwargs={}
        
        data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
        task_sampler = get_task_sampler(
            task_name, n_dims, batch_size, **task_sampler_kwargs
        )
        generating_func = globals()[f"gen_{prompting_strategy}"]
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)
        task = task_sampler()


        all_metrics = []
        print('idhar?')

        break
    break
    #all_metrics[eval_name] = metrics

from models import *
# model = GPT2_Pretrained()
ys = task.evaluate(xs)

bsize, points, dim = xs.shape
ys_b_wide = torch.cat(
    (
        ys.view(bsize, points, 1),
        torch.zeros(bsize, points, dim - 1),
    ),
    axis=2,
)
zs = torch.stack((xs, ys_b_wide),dim=2)
zs = zs.view(bsize, 2 * points, dim)

embeds = model._read_in(zs) # changing into the embedding dimension
output = model._backbone(inputs_embeds=embeds).last_hidden_state
prediction = model._read_out(output)
inds = torch.arange(ys.shape[1])

pred = prediction[:, ::2, 0][:, inds]  # predict only on xs

metrics = task.get_metric()(pred.cpu(), ys)


### trying this on the GPT2_Pretrained() function

model_test = GPT2_Pretrained()
# testing the config class of GPT2-Model
tokenizer = model_test.tokenizer

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize


for item in range(xs.shape[0]): # looping over batch
    prompt = ""
    for example in range(0,xs.shape[1]): # looping over examples
        if example == 0: # first case
            prompt += f"Predict y for x:{xs[item,example,:].tolist()}.\n"
        else:
            xy_pairs = ", ".join(f"(x:{xs[item,j,:].tolist()}, y:{ys[item,j]})" for j in range(0, example))
            prompt += f"{xy_pairs} -> Predict y for x:{xs[item,example,:].tolist()}.\n"
    break

prompt_sent = sent_tokenize(prompt)
test_prompt = prompt_sent[-1]

with open('prompt.txt',"w") as fp:
    fp.write(test_prompt)

# # an outer loop over batch size 
# # an inner loop over number of examples 

# prompt = ""
# for i in range(0,xs_1.shape[0]-1):
#     xy_pairs = ", ".join(f"(x:{xs_1[j,:].tolist()}, y:{ys_1[j]})" for j in range(0, i+1))
#     prompt += f"{xy_pairs} -> Predict y for x_next:{xs_1[i+1,:].tolist()}.\n"





# # input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
# # # outputs = model_test.model.generate(**inputs, max_new_tokens=50) 
# # gen_tokens = model_test.model.generate(
# #     input_ids,
# #     do_sample=True,
# #     temperature = 0.9,
# #     max_length = 1024
# # )
# # prediction = tokenizer.batch_decode(gen_tokens)[0]

# # from transformers import GPT2Tokenizer, GPT2Model
# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # model = GPT2Model.from_pretrained('gpt2')
# # text = "Replace me by any text you'd like."
# # encoded_input = tokenizer(text, return_tensors='pt')
# # generated_ids = model.generate(**encoded_input, max_new_tokens=100, do_sample=True)
# # output = model(**encoded_input)


# # import torch
# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # device = "cuda" if torch.cuda.is_available() else "cpu" # the device to load the model onto

# # model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
# # tokenizer = AutoTokenizer.from_pretrained("gpt2")

# # prompt = "def hello_world():"

# # model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
# # model.to(device)

# # generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
# # tokenizer.batch_decode(generated_ids)[0]
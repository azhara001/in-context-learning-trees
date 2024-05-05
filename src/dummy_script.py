import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
import pandas as pd
import wandb

from train import *
from models import *


parser = QuinineArgumentParser(schema=schema)
args = parser.parse_quinfig()
assert args.model.family in ["gpt2", "lstm"]
print(f"Running with: {args}")

if not args.test_run:
    run_id = args.training.resume_id
    if run_id is None:
        run_id = str(uuid.uuid4())

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir

#model = build_model(args.model)
model = GPT2_Pretrained()


optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
curriculum = Curriculum(args.training.curriculum)
n_dims = 20 # dummy hard coded for now
bsize = args.training.batch_size
data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
task_sampler = get_task_sampler(
    args.training.task,
    n_dims,
    bsize,
    num_tasks=args.training.num_tasks,
    **args.training.task_kwargs,
)

data_sampler_args = {}
task_sampler_args = {}

xs = data_sampler.sample_xs(
    curriculum.n_points,
    bsize,
    curriculum.n_dims_truncated,
    **data_sampler_args,
)
task = task_sampler(**task_sampler_args)
ys = task.evaluate(xs)

zs = model(xs,ys)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
zs = str(zs)
input_ids = tokenizer(zs, return_tensor="pt").input_ids

gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature = 0.9,
            max_length = 100
        )
prediction = tokenizer.batch_decode(gen_tokens)[0]
print(prediction)

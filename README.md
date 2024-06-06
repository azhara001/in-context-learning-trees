This repository contains the code and models for the following project: <br>
**Adapting to Context: A Case Study on In-Context Learning of Decision Tree Algorithms by Large Language Models** as a part of the final project for [INFO 259: Natural Language Processing](https://ucbnlp24.github.io/webpage/) at UC Berkeley 

Note: This repository and the pretrained model check-points for noise-free training are downloaded and built-up on from the following paper: What Can Transformers Learn In-Context? A Case Study of Simple Function Classes and this repository is forked from: (https://github.com/dtsip/in-context-learning)[https://github.com/dtsip/in-context-learning]


## Getting Started 
You can start by cloning the repository and following the steps below. 

1. Install the dependencies for the code using a Conda virtual environment. Note: You would need to adjust the environment YAML file depending on the setup, especially if performing model training in Google Colab

```python
conda env create -f environment.yml
conda activate in-context-learning
```

2. Download the [model checkpoints](https://drive.google.com/drive/folders/1bbA8X_SePC74Nuxu3aDmHwp7coyoo4nn?usp=sharing) and extract them in the current directory.

```
wget https://drive.google.com/drive/folders/1bbA8X_SePC74Nuxu3aDmHwp7coyoo4nn?usp=drive_link.zip
unzip models.zip
```

3. [Optional] If you plan to train, populate ```conf/wandb.yaml``` with your wandb info.

The ```eval.ipynb``` inlcudes the results of the following prompting strategies on decision tree algorithm inference using a GPT2 Config Model
1. Standard prompting - noise free training (checkpoints extracted from [forked repo](https://github.com/dtsip/in-context-learning)
2. Random Quadrant prompting - noise free training (checkpoints extracted from [forked repo](https://github.com/dtsip/in-context-learning)
3. Train-Test overlapping prompting - noise free training (checkpoints extracted from [forked repo](https://github.com/dtsip/in-context-learning)
4. Noisy label training on decision tree algorithms with the following isotropic gaussian noise on the labels:
    - Standard deviation 0
    - Standard deviation 0.05
    - Standard deviation 1
    - Standard deviation 3
  Note: for the following noisy training regime, refer to the custom conf/.. .yaml files in the src directory


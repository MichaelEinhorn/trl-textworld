# Welcome to Transformer Reinforcement Learning (trl)
> Train transformer language models with reinforcement learning.


## What is it?
With `trl` you can train transformer language models with Proximal Policy Optimization (PPO). The library is built on top of the [`transformer`](https://github.com/huggingface/transformers) library by  🤗 Hugging Face. Therefore, pre-trained language models can be directly loaded via `transformers`. At this point only decoder architectures such as GTP2 are implemented.

**Highlights:**
- PPOTrainer: A PPO trainer for language models that just needs (query, response, reward) triplets to optimise the language model.
- GPT2 model with a value head: A transformer model with an additional scalar output for each token which can be used as a value function in reinforcement learning.
- Example: Train GPT2 to generate positive movie reviews with a BERT sentiment classifier.

## How it works
Fine-tuning a language model via PPO consists of roughly three steps:

1. **Rollout**: The language model generates a response or continuation based on query which could be the start of a sentence.
2. **Evaluation**: The query and response are evaluated with a function, model, human feedback or some combination of them. The important thing is that this process should yield a scalar value for each query/response pair.
3. **Optimization**: This is the most complex part. In the optimisation step the query/response pairs are used to calculate the log-probabilities of the tokens in the sequences. This is done with the model that is trained and and a reference model, which is usually the pre-trained model before fine-tuning. The KL-divergence between the two outputs is used as an additional reward signal to make sure the generated responses don't deviate to far from the reference language model. The active language model is then trained with PPO.

## Changes from trl repo
Major
Integrated text world repo for text adventure with trl repo for PPO.
Separate Value Head from LM. Fixes model loading errors and makes it far easier to swap models. Tested with GPT2, GPT2-XL, GPT-Neo-1.3B, GPT-Neox-20B. Any HuggingFace model that can output hidden states should work.
KL divergence term with exact gradients on the current model. Importance sampling corrected for the distribution of the last model.
Implemented Rejection Sampling. Either keep top % or top N trajectories.
Add examples of valid or correct actions to the context window. Both allows the model to explore more states before learning how to solve the first one and gives it an example it can use.
About 3 prior states and actions can fit inside the context window, but this significantly increases RAM Requirements.
Optimizations
Pytorch Lightning Deepspeed for PPO, and Rejection Sampling. Enables fitting much larger models and larger batch sizes into the same GPU RAM. Tested up to 8 GPUs.
Batching for both the environment and text gen forward passes in gameplay. Batching for all Forward and backward passes in PPO. Loss calculations from the model outputs are still per sample.
Caching logprobs and values from gameplay loop. Eliminates 1 forward pass over every batch at beginning of training loop. Still need to do a reference model forward pass over all data in this spot.
Don’t save reference model and optimizer state with pytorch lightning. Was using over 20 GB and several minutes for each saved checkpoint.
Minor
Added a field for next value to allow discounting across multiple experiences in the same game. Separate discount rate for cross token and cross step discounting.
Changed format to python files instead of notebooks.
Changed control flow from textworld to have trainer run the gameplay loop instead of the gameplay loop running training.
String buffer memory that puts previous states and actions into the current context.
Change prompt formats. State, info, You <output action> causes the model to repeat the sentence “You are carrying nothing” from the info. “What do you do?” appears to work better, but the model is not selecting one of the commands in the list.
Future
Tune KL value. Too high and the model learns to output empty strings. At zero the model collapses to spamming the word shape. These results were trained on a reward function of the number of Es within a maximum of 20 tokens.
Add Forward KL
Use only the last 3rd of the model for training and take the intermediate output of reference model as input. Could save GPU Ram and decrease training time.
Use more complex games, larger models, chain of thought prompting, and GEM.
Still running out of RAM for larger context lengths. Tune Deepspeed params, and batch size to fix.

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

### Language models
The language models utilize the `transformers` library by 🤗 Hugging Face.

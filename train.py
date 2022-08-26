import torch
import os
from glob import glob
import gym
import textworld.gym

from ppoValHead import PPOTrainer
from torchinfo import summary

import deepspeed

def getEnvs(download=True):
    if download:
        os.system("wget https://aka.ms/textworld/notebooks/data.zip")
        os.system("unzip -nq data.zip && rm -f data.zip")
    else:
        # Same as !make_games.sh
        os.system("tw-make tw-simple --rewards dense    --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsDense_goalDetailed.z8")
        os.system("tw-make tw-simple --rewards balanced --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsBalanced_goalDetailed.z8")
        os.system("tw-make tw-simple --rewards sparse   --goal detailed --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalDetailed.z8")
        os.system("tw-make tw-simple --rewards dense    --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsDense_goalBrief.z8")
        os.system("tw-make tw-simple --rewards balanced --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsBalanced_goalBrief.z8")
        os.system("tw-make tw-simple --rewards sparse   --goal brief    --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalBrief.z8")
        os.system("tw-make tw-simple --rewards sparse   --goal none     --seed 18 --test --silent -f --output games/tw-rewardsSparse_goalNone.z8")
        

def play(agent, path, max_step=100, nb_episodes=10, verbose=True):
    torch.manual_seed(20211021)  # For reproducibility when using action sampling.

    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    
    gamefiles = [path]
    if os.path.isdir(path):
        gamefiles = glob(os.path.join(path, "*.z8"))
        
    print(gamefiles)
    env_id = textworld.gym.register_games(gamefiles,
                                          request_infos=infos_to_request,
                                          max_episode_steps=max_step)
    print(env_id)
    env = gym.make(env_id)  # Create a Gym environment to play the text game.
    if verbose:
        if os.path.isdir(path):
            print(os.path.dirname(path), end="")
        else:
            print(os.path.basename(path), end="")
        
    # Collect some statistics: nb_steps, final reward.
    avg_moves, avg_scores, avg_norm_scores = [], [], []
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.

        score = 0
        done = False
        nb_moves = 0
        while not done:
            command = agent.act(obs, score, done, infos)
            obs, score, done, infos = env.step(command)
            if hasattr(agent, 'reportScore'):
                agent.reportScore(score, done, infos)
            nb_moves += 1
        
        agent.act(obs, score, done, infos)  # Let the agent know the game is done.
                
        if verbose:
            print(".", end="")
        avg_moves.append(nb_moves)
        avg_scores.append(score)
        avg_norm_scores.append(score / infos["max_score"])

    env.close()
    if verbose:
        if os.path.isdir(path):
            msg = "  \tavg. steps: {:5.1f}; avg. normalized score: {:4.1f} / {}."
            print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
        else:
            msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))
            
def train(model_name, low_ram=True, single_game=True):
    from agents import NLPAgent
    from time import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # gpt2 and gpt2-xl
    if 'gpt2' in model_name:
        from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model_ref = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif model_name == 'gptj':
        from transformers import GPT2Tokenizer, GPTJForCausalLM
        if low_ram:
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
            model_ref = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
            model_ref = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print(model.config.torch_dtype)
    
    agent = NLPAgent(model, model_ref, tokenizer, humanTurns=0)
    print(agent.ppo_trainer.valueHead)
    summary(model)

    model = model.to(device)
    model_ref = model_ref.to(device)
    
    if single_game:
        print("Training")
        agent.train()  # Tell the agent it should update its parameters.
        starttime = time()
        play(agent, "./games/tw-rewardsDense_goalDetailed.z8", nb_episodes=500, verbose=False)  # Dense rewards game.
        print("Trained in {:.2f} secs".format(time() - starttime))
    
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(agent, 'checkpoints/agent_trained_on_single_game.pt')
        
    else:
        print("Training on 100 games")
        agent.train()  # Tell the agent it should update its parameters.
        starttime = time()
        play(agent, "./training_games/", nb_episodes=100 * 5, verbose=False)  # Each game will be seen 5 times.
        print("Trained in {:.2f} secs".format(time() - starttime))
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(agent, 'checkpoints/agent_trained_on_multiple_games.pt')
    
    
if __name__ == "__main__":
    import argparse
    
    getEnvs()
    print("generated envs")
    
    model_name = 'gpt2-xl'
    # model_name = 'gptj'
    low_ram = True
    single_game = True

    train(model_name, low_ram, single_game)
    


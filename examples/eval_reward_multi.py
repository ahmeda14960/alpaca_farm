import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer
import transformers

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Literal
import contextlib
import os
import pathlib
from datasets import load_dataset
#import torch.cuda.blas as blas


from alpaca_farm import common, constants, data_utils, logging
from alpaca_farm.data_utils import DataCollatorForBinaryRewardModelingDataset
from alpaca_farm.models.reward_model import (
    MultiHeadRewardModel,
    RewardModel,
    RewardConfig,
)  # Adjust the import based on where your module is saved
from transformers.trainer_utils import EvalPrediction

import tqdm
import datetime


def create_dataloader(packaged_data, collator, tokenizer, batch_size=32, shuffle=False):
    dataloader = DataLoader(
        packaged_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collator
    )
    return dataloader


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=True,
        metadata={
            "help": "Whether to initialize the model on CPU. "
            "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
            "Ending with EOS might help the reward model realize it's time to predict."
        },
    )
    resume_from_checkpoint: bool = field(
        default=False, metadata={"help": "If True, loads from last check point."}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: Literal[
        "alpaca_human_preference",
        "alpaca_gpt4_preference",
        "alpaca_noisy_multi_preference",
        "stanfordnlp/SHP",
    ] = field(
        default="alpaca_noisy_multi_preference",
        metadata={
            "help": "Name of the dataset. Fetches the human or GPT-4 preference data."
        },
    )
    eval_size: int = field(
        default=0,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    # TODO: Automatically set based on dataset_name.
    prompt_dict_path: str = field(
        default=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )


def main():
    parser = transformers.HfArgumentParser((TrainingArguments, DataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    model_dirs = [
        f"/lfs/skampere1/0/ahmedah/logs/ppo_multi_alp_{run_number}/"
        for run_number in range(2)
    ]
    # model_dir = f"//home/azureuser/out/alp_rw_opt_20231117001420"
    model_dir = f"/lfs/skampere1/0/ahmedah/logs/opt1bmultirwlalp/"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/opt-1.3b",
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=False,
    )

    # use an eval mode that loads validation split, and puts everything on
    # training dataset from validation split
    data_module = data_utils.make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        eval=True,
    )
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    validation_dataloader = create_dataloader(
        train_dataset, data_collator, tokenizer, batch_size=32, shuffle=False
    )

    multi_head = True
    for mode in ("min", "max", "median"):
        average_likelihoods, mean_accuracies, std_accuracies = plot_accuracy_vs_likelihood(model_dir, validation_dataloader, mode, multi_head)
        er_label = "mode = " + mode
        plt.errorbar(average_likelihoods, mean_accuracies, yerr=std_accuracies, fmt='-o', capsize=5, label=er_label)
    plt.xlabel("Likelihood")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Likelihood")
    plt.grid(True)
    plt.legend(loc='upper left')
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    plt.savefig(f"all_new_plot_multi{current_date}.png")
    plt.show()

def accuracy_per_likelihood_bin(model, device, validation_loader, bin_edges, mode="min"):
    assert mode in ["max", "median", "min"], "Invalid mode. Choose from 'max', 'median', or 'min'."

    likelihoods = []
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in tqdm.tqdm(validation_loader, desc="Validating", leave=False):
            batch_size, num_pairs, context_len = batch["input_ids"].shape
            
            all_probs = torch.zeros((batch_size, 1)).cuda(device) # Store all model probabilities
            head_probs = []
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=batch_input_ids.view(batch_size*num_pairs, context_len), 
                            attention_mask=batch_attention_mask.view(batch_size*num_pairs, context_len))
            
            # drop last element which isn't learned. This is due to the hacky way 
            # we train the multi head RM we just ignore the last linear param
            outputs.rewards = outputs.rewards[:-1]
            outputs.rewards = outputs.rewards.view(-1, batch_size, num_pairs)
            exp_rewards = torch.exp(outputs.rewards)
            sum_exp_rewards = exp_rewards.sum(dim=2)
            head_probs = exp_rewards[:, :, 0] / sum_exp_rewards

            head_probs = head_probs.transpose(0, 1)
            torch.cuda.synchronize(device)

            if mode == "max":
                final_probs = head_probs.max(dim=1).values
            elif mode == "median":
                final_probs = head_probs.median(dim=1).values
            else:  # mode == "min"
                final_probs = head_probs.min(dim=1).values

            likelihoods.extend(final_probs.tolist())
            predictions.extend((final_probs <= 0.5).long().tolist())
            ground_truths.extend(batch["choice"].flatten().tolist())

    bin_idxs = np.digitize(likelihoods, bin_edges)
    bin_accuracies = {}
    # since np bin starts 1 indexed, we need to start at 1
    for bin_idx in range(1, len(bin_edges)):
        idxs = [i for i, b in enumerate(bin_idxs) if b == bin_idx]
        if not idxs:
            continue
        correct = sum(predictions[i] == ground_truths[i] for i in idxs)
        bin_accuracies[bin_idx] = correct / len(idxs)

    return bin_accuracies

def plot_accuracy_vs_likelihood(model_dirs, validation_loader, mode, multi_head=False):
    bin_edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # this is zero index, np bin returns 1 index!
    accuracies_by_bin = {i:[] for i in range(1, len(bin_edges))}
    average_likelihoods = []

    device = torch.device('cuda')
    config = RewardConfig.from_pretrained(model_dirs)
    model = MultiHeadRewardModel.from_pretrained(model_dirs, config=config, flash_attn=True, bf16=True)
    model = model.to(device).half().eval()
    # for idx, model_dir in tqdm.tqdm(enumerate(model_dirs), desc="Loading models", leave=False):
    #     config = RewardConfig.from_pretrained(model_dir)
    #     if not multi_head:
    #         model = RewardModel.from_pretrained(model_dir, config=config, flash_attn=True, bf16=True)
    #     else:
    #         model = MultiHeadRewardModel.from_pretrained(model_dir, config=config, flash_attn=True, bf16=True)
    #     model = model.to(device).half().eval()
    #     models.append(model)
    #     device_list.append(device)

    # mode = "min"
    bin_accuracies = accuracy_per_likelihood_bin(model, device, validation_loader, bin_edges, mode=mode)
    for bin_idx, accuracy in bin_accuracies.items():
        # Add accuracy to the correct bin
        accuracies_by_bin[bin_idx].append(accuracy)

        # Compute average likelihood for this bin
        average_likelihood = (bin_edges[bin_idx-1] + bin_edges[bin_idx]) / 2
        print('bin_idx', bin_idx, 'average_likelihood', average_likelihood)
        average_likelihoods.append(average_likelihood)

    # Compute mean and std for accuracies in each bin
    
    mean_accuracies = [np.mean(accuracies_by_bin[i]) for i in range(1, len(bin_edges))]
    std_accuracies = [np.std(accuracies_by_bin[i]) for i in range(1, len(bin_edges))]
    # Plot with error bars
    # (why?)
    # if mode == "min":
    #     mean_accuracies = mean_accuracies[:-1]
    #     std_accuracies = std_accuracies[:-1]
    # import ipdb; ipdb.set_trace()
    return average_likelihoods, mean_accuracies, std_accuracies

if __name__ == "__main__":
    main()

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
    RewardModel,
    RewardConfig,
)  # Adjust the import based on where your module is saved
from transformers.trainer_utils import EvalPrediction

import tqdm


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
        default=2048,
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
        default="stanfordnlp/SHP",
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
        default=pathlib.Path(__file__).parent / "prompts" / "v0_SHP.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )


def main():
    parser = transformers.HfArgumentParser((TrainingArguments, DataArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    # Example usage:
    dataset = load_dataset("stanfordnlp/SHP", split="validation")

    model_dirs = [
        f"~/logs/debug-shp-rwl1b-{run_number}/"
        for run_number in range(1, 6)
    ]

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
        train_dataset, data_collator, tokenizer, batch_size=8, shuffle=False
    )

    plot_accuracy_vs_likelihood(model_dirs, validation_dataloader)

def accuracy_per_likelihood_bin(models, device_list, validation_loader, bin_edges):
    likelihoods = []
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in tqdm.tqdm(validation_loader, desc="Validating", leave=False):
            batch_size, num_pairs, context_len = batch["input_ids"].shape
            max_probs = torch.zeros((batch_size,)).cuda(device_list[0])  # Initialize on first GPU


            for idx, model in tqdm.tqdm(enumerate(models), desc="Ensembling", leave=False):
                device = device_list[idx]
                #max_probs = max_probs.to(device)
                batch_input_ids = batch["input_ids"].to(device)
                batch_attention_mask = batch["attention_mask"].to(device)
                outputs = model(batch_input_ids.view(batch_size*num_pairs, context_len), 
                                batch_attention_mask.view(batch_size*num_pairs, context_len))
                outputs.rewards = outputs.rewards.view(batch_size, num_pairs)

                exp_rewards = torch.exp(outputs.rewards)
                sum_exp_rewards = exp_rewards.sum(dim=1)
                prob_choice_0 = exp_rewards[:, 0] / sum_exp_rewards
                max_probs = torch.where(prob_choice_0 > max_probs, prob_choice_0, max_probs)
                torch.cuda.synchronize(device)  # Wait for all CUDA operations to complete


            likelihoods.extend(max_probs.tolist())
            predictions.extend((max_probs <= 0.5).long().tolist())
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

def plot_accuracy_vs_likelihood(model_dirs, validation_loader):
    bin_edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # this is zero index, np bin returns 1 index!
    accuracies_by_bin = {i:[] for i in range(1, len(bin_edges))}
    average_likelihoods = []

    models = []
    device_list = []
    device = torch.device('cuda')
    for idx, model_dir in tqdm.tqdm(enumerate(model_dirs), desc="Loading models", leave=False):
        config = RewardConfig.from_pretrained(model_dir)
        model = RewardModel.from_pretrained(model_dir, config=config, flash_attn=True, bf16=True)
        model = model.to(device).half().eval()
        models.append(model)
        device_list.append(device)

    bin_accuracies = accuracy_per_likelihood_bin(models, device_list, validation_loader, bin_edges)
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
    plt.errorbar(average_likelihoods, mean_accuracies, yerr=std_accuracies, fmt='-o', capsize=5)
    plt.xlabel("Likelihood")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Likelihood")
    plt.grid(True)
    plt.savefig("new_plot.png")
    plt.show()

if __name__ == "__main__":
    main()

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
        f"/scr-ssd/ahmedah/models/opt1b-rwl-shp-{run_number}/"
        for run_number in range(0, 6)
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
        train_dataset, data_collator, tokenizer, batch_size=4, shuffle=False
    )

    plot_accuracy_vs_likelihood(model_dirs, validation_dataloader)


def accuracy_per_likelihood_bin(model, validation_loader, bin_edges, device):
    """Compute accuracy for samples within different likelihood bins."""
    likelihoods = []
    predictions = []
    ground_truths = []
    print(device)
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validating", leave=False):
            # Move the batch data to the appropriate GPU device
            batch_size, num_pairs, context_len = batch["input_ids"].shape

            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            # this causes RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`
            # outputs = model(
            #     batch["input_ids"].view(batch_size * num_pairs, context_len).to(device),
            #     batch["attention_mask"]
            #     .view(batch_size * num_pairs, context_len)
            #     .to(device),
            # )
            outputs = model(batch["input_ids"].view(batch_size*num_pairs, context_len), batch["attention_mask"].view(batch_size*num_pairs, context_len))
            outputs.rewards = outputs.rewards.view(batch_size, num_pairs)

            # Compute the Bradley-Terry probabilities
            exp_rewards = torch.exp(outputs.rewards)

            # Calculate the sum of exp rewards for each sample (which will be used as a denominator)
            sum_exp_rewards = exp_rewards.sum(dim=1)
            # Compute the Bradley-Terry probability for choice[0]
            prob_choice_0 = exp_rewards[:, 0] / sum_exp_rewards
            batch_likelihoods = prob_choice_0.tolist()

            # Predictions are 0 (choice[0]) if prob_choice_0 > 0.5 else 1 (choice[1])
            batch_predictions = (prob_choice_0 <= 0.5).long().tolist()

            likelihoods.extend(batch_likelihoods)
            predictions.extend(batch_predictions)
            ground_truths.extend(batch["choice"].tolist())

    # Bin the likelihoods
    bin_idxs = np.digitize(likelihoods, bin_edges)
    bin_accuracies = {}
    for bin_idx in range(len(bin_edges) + 1):
        idxs = [i for i, b in enumerate(bin_idxs) if b == bin_idx]
        if not idxs:
            continue

        correct = sum(predictions[i] == ground_truths[i] for i in idxs)
        bin_accuracies[bin_idx] = correct / len(idxs)

    return bin_accuracies


def plot_accuracy_vs_likelihood(model_dirs, validation_loader):
    # Define the likelihood bins
    bin_edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_accuracies = []
    average_likelihoods = []

    # Load all models onto their respective GPUs
    models = []
    device_list = []
    for idx, model_dir in enumerate(model_dirs):
        if idx != 0:
            continue
        device = torch.device(
            f"cuda:{idx % 7}"
        )  # Ensures distribution across cuda:0 to cuda:6
        config = RewardConfig.from_pretrained(model_dir)
        model = RewardModel.from_pretrained(
            model_dir, config=config, flash_attn=True, bf16=True
        )
        model = model.to(device).half().eval()
        models.append(model)
        device_list.append(device)

    # Process the validation data through the entire ensemble and compute bin accuracies
    for idx, model in enumerate(models):
        bin_accuracies = accuracy_per_likelihood_bin(
            model, validation_loader, bin_edges, device_list[idx]
        )

        for bin_idx, accuracy in bin_accuracies.items():
            model_accuracies.append(accuracy)
            # Average likelihood calculation remains unchanged
            if bin_idx == 0:
                average_likelihood = bin_edges[bin_idx] / 2
            else:
                average_likelihood = (bin_edges[bin_idx - 1] + bin_edges[bin_idx]) / 2
            average_likelihoods.append(average_likelihood)

    # Plotting
    plt.scatter(average_likelihoods, model_accuracies)
    plt.xlabel("Likelihood")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Likelihood")
    plt.grid(True)
    plt.savefig("plot.png")
    plt.show()


if __name__ == "__main__":
    main()

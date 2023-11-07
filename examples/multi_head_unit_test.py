# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import pathlib
from dataclasses import dataclass, field
from typing import Literal

import transformers
import torch.nn.functional as F
import einops
import torch

from alpaca_farm import common, constants, data_utils, logging, torch_ops
from alpaca_farm.models import reward_model
from alpaca_farm.reward_modeling_trainer import  EnsembleTrainer, compute_multi_reward_modeling_metrics
from multi_reward_modeling import ModelArguments, DataArguments, TrainingArguments
logger = logging.get_logger(__name__)

class TestRewardModel():
    def __init__(self, num_heads: int = 4):
        self.backbone_model = torch.nn.Linear()
        hidden_size = 
        reward_head = torch.nn.ModuleList([torch.nn.Linear(hidden_size, 1) for i in range(num_heads)])
        for i in range(num_heads):
            torch.nn.init.zeros_(reward_head[i].bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

    def forward(self, input_ids, head_index, attention_mask=None, return_dict=True):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        outputs = self.backbone_model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        # TODO(lxuechen): Make returning rewards at all positions and last_hidden_state an option.
        rewards = self.reward_head[head_index](last_hidden_state_at_the_end).squeeze(-1)
        return rewards
    
    def compute_training_loss(self, inputs):
        losses = []
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise
        input_ids, attention_mask, choice = common.unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "choice")
        )
        num_candidates = input_ids.size(1)
        
        num_per_head = input_ids.size(0)//self.num_heads
        # import ipdb; ipdb.set_trace()
        for i in range(self.num_heads):
            split_per_head = torch.randint(low=0, high=input_ids.size(0), size=(num_per_head,))
            input_ids_i = input_ids[split_per_head]
            attention_mask_i = attention_mask[split_per_head]
            choice_i = choice[split_per_head]

            input_ids_flat, attention_mask_flat = tuple(
                einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids_i, attention_mask_i)
            )
            
            outputs = self.forward(input_ids=input_ids_flat, head_index=i, attention_mask=attention_mask_flat)
            rewards_flat = outputs.rewards
            rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)  # Size: (bsz, num_candidates).

            logits = rewards

            loss_head = F.binary_cross_entropy_with_logits(logits, choice_i.to(logits.dtype), reduction="mean")
            print("----Working with head " + str(i) + " ----")
            for j in range(self.num_heads):
                loss_j = torch.autograd.grad(loss_head, self.reward_head[j].weight, allow_unused=True)
                if i == j:
                    print("Loss for head " + str(j) + "should be non-zero: " + str(loss_j))
                else:
                    print("Loss for head " + str(j) + "should be 0: " + str(loss_j))
            losses.append(loss_head)

        # Average the losses from all heads
        loss = sum(losses) / self.num_heads
        return loss

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    _, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.prompt_dict_path = pathlib.Path(__file__).parent / "prompts" / "v0_SHP.json" if "SHP" in data_args.dataset_name else pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/opt-1.3b",
        cache_dir=constants.DEFAULT_CACHE_DIR,
        model_max_length=512,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=False,
    )
    tokenizer.padding = training_args.padding
    data_module = data_utils.make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    input_ids, attention_mask, choice = 
    new_model = TestRewardModel()
    inputs = {}
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    inputs["choice"] = choice
    return new_model.compute_training_loss(inputs)


if __name__ == "__main__":
    main()
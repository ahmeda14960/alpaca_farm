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

class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self._name_or_path = backbone_model_name_or_path

class MultiHeadRewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig

    def __init__(self, config: RewardConfig, num_heads: int = 4, **kwargs):
        super(MultiHeadRewardModel, self).__init__(config)
        self.backbone_model = common.make_generative_lm(config.backbone_model_name_or_path, **kwargs)
        self.backbone_model = self.backbone_model.to(torch.bfloat16)
        hidden_size = common.get_transformer_hidden_size(self.backbone_model)
        reward_head = torch.nn.ModuleList([torch.nn.Linear(hidden_size, 1) for i in range(num_heads)])
        for i in range(num_heads):
            torch.nn.init.zeros_(reward_head[i].bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

    def forward(self, input_ids, head_index, attention_mask=None, return_dict=True, **kwargs):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        outputs = self.backbone_model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        # TODO(lxuechen): Make returning rewards at all positions and last_hidden_state an option.
        rewards = self.reward_head[head_index](last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)
    
def compute_training_loss(inputs, model, num_heads):
    losses = []
    # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
    # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
    # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise
    input_ids, index_0, index_1, choice = common.unpack_dict(
        inputs, keys=("input_ids", "index_0", "index_1", "choice")
    )
    print(input_ids.shape)
    
    num_candidates = input_ids.size(1)
    
    num_per_head = input_ids.size(0)//num_heads
    # import ipdb; ipdb.set_trace()
    for i in range(num_heads):
        # split_per_head = torch.randint(low=0, high=input_ids.size(0), size=(num_per_head,))
        # input_ids_i = input_ids[split_per_head]
        # attention_mask_i = attention_mask[split_per_head]
        # index_0_i = index_0[split_per_head]
        # index_1_i = index_1[split_per_head]
        # choice_i = choice[split_per_head]

        # input_ids_flat, attention_mask_flat = tuple(
        #     einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        # )
        
        attention_mask = (torch.triu(torch.ones(inputs["input_ids"].shape[1], inputs["input_ids"].shape[0])) == 1).transpose(0, 1)
        attention_mask = attention_mask.bfloat16().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))

        print("HELLO")
        outputs = model.forward(input_ids=input_ids, head_index=i, attention_mask=attention_mask)
        print("GOODBYE")
        logits = outputs.rewards
        # rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)  # Size: (bsz, num_candidates).

        # rewards_0, rewards_1 = tuple(
        #     torch_ops.batch_select(rewards, index) for index in (index_0, index_1)
        # )  # Size: (bsz, num_pairs).
        # logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).

        loss_head = F.binary_cross_entropy_with_logits(logits, choice.to(logits.dtype), reduction="mean")
        print("----Working with head " + str(i) + " ----")
        for j in range(num_heads):
            loss_j = torch.autograd.grad(loss_head, model.reward_head[j].weight, allow_unused=True)
            if i == j:
                print("Loss for head " + str(j) + "should be non-zero: " + str(loss_j))
            else:
                print("Loss for head " + str(j) + "should be 0: " + str(loss_j))
        losses.append(loss_head)

    # Average the losses from all heads
    loss = sum(losses) / num_heads
    return loss

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    _, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/opt-1.3b",
        cache_dir=constants.DEFAULT_CACHE_DIR,
        model_max_length=512,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=False,
        padding = True,
        max_length=512,
    )
    data_args.prompt_dict_path = pathlib.Path(__file__).parent / "prompts" / "v0_SHP.json" if "SHP" in data_args.dataset_name else pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json"
    data_module = data_utils.make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    # import ipdb; ipdb.set_trace()
    train_dataloader = torch.utils.data.DataLoader(data_module["train_dataset"], batch_size=4, shuffle=True)
    dataiter = iter(train_dataloader)
    mid_inputs = next(dataiter)
    num_heads = 4
    rew_config = RewardConfig(backbone_model_name_or_path="//home/azureuser/out/opt_1b_alpsft_20231116213715")
    new_model = MultiHeadRewardModel(config=rew_config, num_heads=num_heads, flash_attn=True, fp16=False, bf16=True, low_cpu_mem_usage=True, device_map=None)
    inputs = {}
    # import ipdb; ipdb.set_trace()
    print(mid_inputs)
    inputs["input_ids"] = mid_inputs["input_ids"][0]
    inputs["choice"] = mid_inputs["choice"][0]
    inputs["index_0"] = mid_inputs["index_0"][0]
    inputs["index_1"] = mid_inputs["index_1"][0]
    return compute_training_loss(inputs, new_model, num_heads)


if __name__ == "__main__":
    main()
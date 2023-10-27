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
from dataclasses import dataclass

import transformers
import einops
import torch

from alpaca_farm import common, data_utils, logging, torch_ops
from alpaca_farm.models import reward_model
from alpaca_farm.reward_modeling_trainer import  EnsembleTrainer, compute_multi_reward_modeling_metrics
from multi_reward_modeling import ModelArguments, DataArguments, TrainingArguments
logger = logging.get_logger(__name__)


@dataclass
class TestEnsembleTrainer(EnsembleTrainer):
    def compute_training_loss(self, model, inputs, return_outputs=False):
        
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise
        input_ids, attention_mask, index_0, index_1, choice = common.unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "index_0", "index_1", "choice")
        )
        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        
        num_per_head = input_ids.size(0)//self.num_heads
        # import ipdb; ipdb.set_trace()
        for i in range(self.num_heads):
            split_per_head = torch.randint(low=0, high=input_ids.size(0), size=(num_per_head,))
            input_ids_i = input_ids[split_per_head]
            attention_mask_i = attention_mask[split_per_head]
            index_0_i = index_0[split_per_head]
            index_1_i = index_1[split_per_head]
            choice_i = choice[split_per_head]

            input_ids_flat, attention_mask_flat = tuple(
                einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids_i, attention_mask_i)
            )
            
            outputs = model(input_ids=input_ids_flat, head_index=i, attention_mask=attention_mask_flat)
            rewards_flat = outputs.rewards
            rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)  # Size: (bsz, num_candidates).

            rewards_0, rewards_1 = tuple(
                torch_ops.batch_select(rewards, index) for index in (index_0_i, index_1_i)
            )  # Size: (bsz, num_pairs).
            logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).

            loss_head = F.binary_cross_entropy_with_logits(logits, choice_i.to(logits.dtype), reduction="mean")
            loss_head.backward()
            print("----Working with head " + i + " ----")
            for j in range(self.num_heads):
                loss_j = torch.autograd.grad(loss_head, model.reward_head[j].weight, allow_unused=True)
                if i == j:
                    print("Loss for head " + j + "should be non-zero: " + loss_j)
                else:
                    print("Loss for head " + j + "should be 0: " + loss_j)

 


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    print(data_args)
    print(training_args)
    if training_args.deepspeed is not None:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = None
    elif training_args.initialize_model_on_cpu:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = True
    else:
        ctx_mgr = common.staggered_object_creation(
            local_rank=training_args.local_rank, world_size=training_args.world_size
        )
        device_map = {"": training_args.device.index}
        low_cpu_mem_usage = True

    with ctx_mgr:
        config = reward_model.RewardConfig(
            backbone_model_name_or_path=model_args.model_name_or_path
        )
        model = reward_model.MultiHeadRewardModel(
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            config=config,
        )
        common.let_model_save_mem_when_zero_grad(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/opt-1.3b",
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",  # Ensure reward is always extracted at the last token embedding.
        use_fast=training_args.use_fast_tokenizer,
    )
    tokenizer.padding = training_args.padding
    data_args.prompt_dict_path = pathlib.Path(__file__).parent / "prompts" / "v0_SHP.json" if "SHP" in data_args.dataset_name else pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json"
    data_module = data_utils.make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    trainer = TestEnsembleTrainer(
        num_heads=4,  # Number of ensemble members (you can adjust this)
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_multi_reward_modeling_metrics, 
        **data_module,
    )

    trainer.train()


if __name__ == "__main__":
    main()
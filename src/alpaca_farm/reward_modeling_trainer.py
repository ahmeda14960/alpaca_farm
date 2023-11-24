# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_sagemaker_mp_enabled,
)

if is_sagemaker_mp_enabled():
    from .trainer_pt_utils import smp_forward_backward

from alpaca_farm import common, torch_ops


class Trainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        input_ids, attention_mask, index_0, index_1, choice = common.unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "index_0", "index_1", "choice")
        )
        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        )
        outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        rewards_flat = outputs.rewards
        rewards = einops.rearrange(
            rewards_flat, "(b c) -> b c", c=num_candidates
        )  # Size: (bsz, num_candidates).

        rewards_0, rewards_1 = tuple(
            torch_ops.batch_select(rewards, index) for index in (index_0, index_1)
        )  # Size: (bsz, num_pairs).
        logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).
        # Type casting of `choice` is due to amp.autocast context manager.
        if len(choice.shape) > 1 and logits.shape != choice.shape:
            choice = choice.squeeze(-1)
        # squeeze to avoid error with SHP from (bsz, 1)
        loss = F.binary_cross_entropy_with_logits(
            logits, choice.to(logits.dtype), reduction="mean"
        )
        return (loss, dict(logits=logits)) if return_outputs else loss


# Take a batch, pass it through one head and make sure through backprop there is no grad updates
# to other heads. 
class EnsembleTrainer(transformers.Trainer):
    def __init__(self, num_heads=4, *args, **kwargs):
        self.num_heads = num_heads
        #self.create_accelerator_and_postprocess()
        #import ipdb; ipdb.set_trace()
        super().__init__(*args, **kwargs)

    def compute_training_loss(self, model, inputs, return_outputs=False):
        losses = []
        logits_list = []
         
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
            # split_per_head = torch.randint(low=0, high=input_ids.size(0), size=(num_per_head,))
            # input_ids_i = input_ids[split_per_head]
            # attention_mask_i = attention_mask[split_per_head]
            # index_0_i = index_0[split_per_head]
            # index_1_i = index_1[split_per_head]
            # choice_i = choice[split_per_head]

            input_ids_flat, attention_mask_flat = tuple(
                einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
            )
            
            outputs = model(input_ids=input_ids_flat, head_index=i, attention_mask=attention_mask_flat)
            rewards_flat = outputs.rewards
            rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)  # Size: (bsz, num_candidates).

            rewards_0, rewards_1 = tuple(
                torch_ops.batch_select(rewards, index) for index in (index_0, index_1)
            )  # Size: (bsz, num_pairs).
            logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).

            loss_head = F.binary_cross_entropy_with_logits(logits, choice.to(logits.dtype), reduction="mean")

            losses.append(loss_head)
            logits_list.append(logits)

        # Average the losses from all heads
        loss = sum(losses) / self.num_heads

        if return_outputs:
            return (loss, dict(logits=logits_list))  # Return logits from all heads
        else:
            return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        losses = []
        logits_list = []
        
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
            input_ids_flat, attention_mask_flat = tuple(
                einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
            )
            
            outputs = model(input_ids=input_ids_flat, head_index=i, attention_mask=attention_mask_flat)
            rewards_flat = outputs.rewards
            rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)  # Size: (bsz, num_candidates).

            rewards_0, rewards_1 = tuple(
                torch_ops.batch_select(rewards, index) for index in (index_0, index_1)
            )  # Size: (bsz, num_pairs).
            logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).

            loss_head = F.binary_cross_entropy_with_logits(logits, choice.to(logits.dtype), reduction="mean")

            losses.append(loss_head)
            logits_list.append(logits)

        # Average the losses from all heads
        loss = sum(losses) / self.num_heads
        # logits = sum(logits_list) / self.num_heads
        
        if return_outputs:
            return (loss, dict(logits=logits_list))  # Return logits from all heads
        else:
            return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_training_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()
# # Example usage
# trainer = EnsembleTrainer(
#     model=model,  # Your model
#     data_collator=data_collator,  # Your data collator
#     train_dataset=train_dataset,  # Your training dataset
#     compute_metrics=compute_metrics,  # Your metric function
#     num_heads=5  # Number of ensemble members (you can adjust this)
# )

# # Training loop
# trainer.train()

def compute_multi_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    # import ipdb; ipdb.set_trace()
    logits = torch.tensor(eval_prediction.predictions).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    if logits.dim() == 1: #one-head
        logits = logits.reshape(1, -1)
    #labels = labels.reshape(logits.shape[0], -1)
    labels = labels.view(1,-1).repeat(logits.shape[0], 1)
    metric_dict = {}
    for i in range(logits.shape[0]):
        
        # if logits.dim == 2:
        #     logits = logits.reshape(logits.shape[0]*logits.shape[1])
        logits_head = logits[i]
        labels_head = labels[i]
        # if labels isn't single dim squeeze again
        if len(labels_head.shape) > 1:
            labels_head = labels_head.squeeze(-1)
        predictions = (logits_head >= 0.0).long()
        metric_dict["accuracy_" + str(i)] = predictions.eq(labels_head).float().mean().item()
        metric_dict["label_positive_rate_" + str(i)] = (labels_head == 1).float().mean().item()
        metric_dict["positive_rate_" + str(i)] = (predictions == 1).float().mean().item()
        metric_dict["true_positive_rate_" + str(i)] = (
            predictions * labels_head
        ).float().sum().item() / labels_head.sum().item()
        metric_dict["false_positive_rate_" + str(i)] = (predictions * (1 - labels_head)).float().sum().item() / (
            1 - labels_head
        ).sum().item()
    return metric_dict
    # dict(
    #     accuracy=accuracy,
    #     label_positive_rate=label_positive_rate,
    #     positive_rate=positive_rate,
    #     true_positive_rate=true_positive_rate,
    #     false_positive_rate=false_positive_rate,
    #     dummy=0,
    # )


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(eval_prediction.predictions).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    # if labels isn't single dim squeeze again
    if len(labels.shape) > 1:
        labels = labels.squeeze(-1)
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    positive_rate = (predictions == 1).float().mean().item()
    true_positive_rate = (
        predictions * labels
    ).float().sum().item() / labels.sum().item()
    false_positive_rate = (predictions * (1 - labels)).float().sum().item() / (
        1 - labels
    ).sum().item()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        positive_rate=positive_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )
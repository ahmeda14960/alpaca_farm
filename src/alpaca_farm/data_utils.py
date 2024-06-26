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

import datasets
import pandas as pd
import transformers

from . import logging, utils
from .data_postprocessor import RewardConditioningPromptPostprocessor
from .data_preprocessor import (
    BinaryRewardModelingDataset,
    DataCollatorForBinaryRewardModelingDataset,
    DataCollatorForSFTDataset,
    DataCollatorForStackableDataset,
    QueryDataset,
    QueryResponseDataset,
    SFTDataset,
    split_train_into_train_and_eval,
)

logger = logging.get_logger(__name__)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    if "alpaca" in data_args.dataset_name:
        alpaca_instructions = datasets.load_dataset(
            data_args.dataset_path, data_args.dataset_name
        )
    else:
        alpaca_instructions = datasets.load_dataset(data_args.dataset_name)
        data_args.train_splits = ["train"]
        data_args.eval_splits = ["validation"]
    train_df = pd.concat(
        [pd.DataFrame(alpaca_instructions[split]) for split in data_args.train_splits]
    )
    train_dataset = SFTDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        dataset=data_args.dataset_name,
    )

    eval_dataset = None
    if data_args.eval_splits is not None:
        found_splits = [
            pd.DataFrame(alpaca_instructions[split])
            for split in data_args.eval_splits
            if split in alpaca_instructions
        ]
        if len(found_splits) > 0:
            eval_df = pd.concat(found_splits)
            eval_dataset = SFTDataset(
                df=eval_df,
                prompt_dict=prompt_dict,
                tokenizer=tokenizer,
                dataset=data_args.dataset_name,
                split=data_args.eval_splits[0],
            )

    if eval_dataset is None:
        logger.warning("Didn't find evaluation dataset. Disabling evaluation.")
        training_args.do_eval = False

    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def make_binary_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
    eval=False,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)
    data_collator = DataCollatorForBinaryRewardModelingDataset(tokenizer=tokenizer)

    if "SHP" in data_args.dataset_name:
        alpaca_human_preference = datasets.load_dataset(data_args.dataset_name)
        split = "validation" if eval else "train"
    else:
        alpaca_human_preference = datasets.load_dataset(
            data_args.dataset_path, data_args.dataset_name
        )
        split = "preference"

    train_df = pd.DataFrame(alpaca_human_preference[split])

    train_dataset = BinaryRewardModelingDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        end_sequence_with_eos=training_args.end_sequence_with_eos,
        dataset=data_args.dataset_name,
        split=split,
    )

    if "SHP" in data_args.dataset_name:
        # if we're in eval mode, return the entire validation
        # split as the training df, other wise properly make
        # splits from val and train
        if eval:
            return dict(
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
        else:
            eval_df = pd.DataFrame(alpaca_human_preference["validation"])

            eval_dataset = BinaryRewardModelingDataset(
                df=eval_df,
                prompt_dict=prompt_dict,
                tokenizer=tokenizer,
                end_sequence_with_eos=training_args.end_sequence_with_eos,
                dataset=data_args.dataset_name,
                split="validation",
            )
    else:
        train_dataset, eval_dataset = split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=data_args.eval_size,
            seed=training_args.seed,
        )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def make_rl_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    if "alpaca" in data_args.dataset_name:
        alpaca_instructions = datasets.load_dataset(
            data_args.dataset_path, data_args.dataset_name
        )
    else:
        alpaca_instructions = datasets.load_dataset(data_args.dataset_name)
        data_args.train_splits = ["train"]
        data_args.eval_splits = ["validation"]
    train_df = pd.concat(
        [pd.DataFrame(alpaca_instructions[split]) for split in data_args.train_splits]
    )
    eval_df = pd.concat(
        [pd.DataFrame(alpaca_instructions[split]) for split in data_args.eval_splits]
    )

    if getattr(training_args, "num_reward_tokens", 0) > 0 and not getattr(
        training_args, "train_on_best_quantile", True
    ):
        prompt_postprocessor = RewardConditioningPromptPostprocessor()
    else:
        prompt_postprocessor = None

    train_dataset = QueryDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        dataset=data_args.dataset_name,
        query_len=training_args.query_len,
        prompt_postprocessor=prompt_postprocessor,
        split=data_args.train_splits[0],
    )

    rollouts_dataset = QueryResponseDataset(
            df=train_df,
            tokenizer=tokenizer,
            query_len=training_args.query_len,
            response_len=training_args.response_len,
            prompt_dict=prompt_dict,
            prompt_postprocessor=prompt_postprocessor,
    )
    eval_dataset = QueryDataset(
        df=eval_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        dataset=data_args.dataset_name,
        query_len=training_args.query_len,
        prompt_postprocessor=prompt_postprocessor,
        split=data_args.eval_splits[0],
    )
    return dict(
        train_dataset=rollouts_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForStackableDataset(),
    )

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

import pathlib
import sys
from typing import Dict, Optional, Sequence, Union

import datasets
import fire
import pandas as pd

from alpaca_farm import data_preprocessor, distributed_utils, utils
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath, AnyPathOrNone

sample_mode_formatter = (
    "temperature={temperature},max_new_tokens={max_new_tokens},seed={seed}"
)

def run_rerank(
    list_dict_data_or_path: Union[Sequence[Dict], AnyPath],
    scorer_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
    per_device_batch_size=4,
    rerank_top_k=1,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
    singleton=False,
    dataset_path="tatsu-lab/alpaca_farm",
    dataset_name: Optional[str] = "alpaca_farm_evaluation",
    prompt_dict_path=pathlib.Path(__file__).parent
    / "prompts"
    / "v0_inputs_noinputs.json",
):
    """Rerank sequences with reward model.

    Args:
        list_dict_data_or_path: Sequence of dict data or a path to it.
            Each dict should have the keys 'prompt' and 'completion' with string values that can be added together.
        scorer_name_or_path: Name or path of the reward model.
        output_path: Optional path to save the rerank results.
        per_device_batch_size: Batch size for reranking for each device.
        rerank_top_k: Keep top k among the reranked sequences.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        flash_attn: Turns on flash_attn for the reward model if True.
        singleton: If True, we are reading from a seprate json with generated samples.

    Returns:
        Rerank results as a list of dict data.
    """
    if isinstance(list_dict_data_or_path, (str, pathlib.Path)):
        list_dict_data_or_path = utils.jload(list_dict_data_or_path)

    if singleton:
        dataset = datasets.load_dataset(dataset_path, dataset_name)

        prompts, list_dict_data, metadata = data_preprocessor.format_prompt_with_data_frame(
            df=pd.DataFrame(dataset["eval"]),
            prompt_dict=utils.jload(prompt_dict_path),
            dataset=dataset_name,
        )

        # outputs is just a list of lists, each of which are all the completions for a single prompt
        # aka the "N" in "best of N"
        outputs = [dict_data["output"] for dict_data in list_dict_data_or_path]
        list_dict_data = [
        {
            "instruction": dict_data["instruction"],
            "input": dict_data["input"],
            "output": output,
            "prompt": prompt,
        }
        for dict_data, prompt, output in utils.zip_(list_dict_data_or_path, prompts, outputs)
        ]

        list_dict_data_or_path = list_dict_data
    

    sequences = [
        [dict_data["prompt"] + output for output in dict_data["output"]]
        for dict_data in list_dict_data_or_path
    ]
    # TODO(lxuechen): FlashAttention reward model is not correctly loaded.
    top_sequences, top_indices, bottom_sequences, bottom_indices, aligned_rewards = score.rerank_sequences_with_huggingface(
        sequences=sequences,
        model_name_or_path=scorer_name_or_path,
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        tf32=tf32,
        flash_attn=flash_attn,
        rerank_top_k=rerank_top_k,
        return_rewards=True,
    )

    return_list_dict_data = [
        {
            "instruction": dict_data["instruction"],
            "input": dict_data["input"],
            "output": dict_data["output"],
            "top_sequence": top_sequence,
            "top_index": top_index,
            "bottom_sequence": bottom_sequence,
            "bottom_index": bottom_index,
            "scorer_name_or_path": scorer_name_or_path,
            "rewards": reward_list,
        }
        for top_sequence, top_index, bottom_sequence, bottom_index, reward_list, dict_data in utils.zip_(
            top_sequences, top_indices, bottom_sequences, bottom_indices, aligned_rewards, list_dict_data_or_path
        )
    ]
    if output_path is not None and distributed_utils.is_main_process():
        utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data


def score_n(
    decoder_name_or_path: AnyPath,
    scorer_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
    input_path: AnyPathOrNone = None,
    prompt_dict_path=pathlib.Path(__file__).parent
    / "prompts"
    / "v0_inputs_noinputs.json",
    split="val",
    per_device_batch_size=4,
    max_instances=sys.maxsize,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=300,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
    rerank_top_k=1,
    dump_all=False,
    singleton=False,
):
    """Chain together decoding and rerank."""
    rerank_return_list_dict_data = run_rerank(
        list_dict_data_or_path=input_path,
        scorer_name_or_path=scorer_name_or_path,
        output_path=output_path,
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        tf32=tf32,
        flash_attn=flash_attn,
        rerank_top_k=rerank_top_k,
        singleton=singleton,
    )

    # Convert best-k-of-n into best-of-n.
    return_list_dict_data = [
        {
            "instruction": rerank_dict_data["instruction"],
            "input": rerank_dict_data["input"],
            "output": rerank_dict_data["output"],
            "decoder_name_or_path": decoder_name_or_path,
            "scorer_name_or_path": scorer_name_or_path,
            "top_sequence": rerank_dict_data["top_sequence"],
            "top_index": rerank_dict_data["top_index"],
            "bottom_sequence": rerank_dict_data["bottom_sequence"],
            "bottom_index": rerank_dict_data["bottom_index"],
            "rewards": rerank_dict_data["rewards"],
        }
        for rerank_dict_data in rerank_return_list_dict_data
    ]

    # call to run_rank above already dumps multiple (ranked) completions per prompt to output_path
    # setting dump_all = True prevents that output from being overwritten
    if not dump_all:
        if output_path is not None and distributed_utils.is_main_process():
            utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)

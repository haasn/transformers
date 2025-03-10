#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
from random import randint

from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_PAST = 1023     # Hardcoded context limit for GPT2
BUFFER_SIZE = 200   # How many words to chop off the front of the context

MAX_SEED = 65536

ALL_MODELS = GPT2Config.pretrained_config_archive_map.keys()

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
}

def set_seed(args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, num_samples=1, temperature=1, top_k=0, top_p=0.0,
                    repetition_penalty=1.0, device='cpu', tokenizer=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context.clone().detach()
    prev_generated = generated
    past = None

    with torch.no_grad():
        while True:

            output, past = model(context, past=past)
            next_token_logits = output[:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            context = next_token
            generated = torch.cat((generated, next_token), dim=1)

            eos = False
            for o in next_token.tolist():
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                print(text, end="", flush=True)
                if '.' in text:
                    eos = True

            while eos:
                print()
                raw_text = input('> ')
                if raw_text == 'quit':
                    return
                if raw_text == 'revert':
                    generated = prev_generated
                    context = generated
                    past = None
                    continue

                prev_generated = generated
                eos = False

                if raw_text != '':
                    next_input = tokenizer.encode(' ' + raw_text, add_special_tokens=False)
                    next_input = torch.tensor(next_input, dtype=torch.long, device=device)
                    next_input = next_input.unsqueeze(0).repeat(num_samples, 1)
                    generated = torch.cat((generated, next_input), dim=1)
                    context = generated
                    past = None


            if past and past[0].size()[3] > MAX_PAST:
                past = None
                context_len = MAX_PAST - BUFFER_SIZE
                context_start = generated.size()[1] - context_len
                context = torch.narrow(generated, 1, context_start, context_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=None,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if args.seed:
        set_seed(args, args.seed)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    logger.info(args)

    while True:

        if args.seed is None:
            new_seed = randint(0, MAX_SEED)
            logger.info('New seed: ' + str(new_seed))
            set_seed(args, new_seed)

        if args.prompt:
            raw_text = args.prompt
            print('>>>', raw_text)
        else:
            raw_text = input(">>> ")

        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)

        try:
            sample_sequence(
                model=model,
                context=context_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=args.device,
                tokenizer=tokenizer,
            )
        except KeyboardInterrupt:
            print("\nInterrupted!\n")

        if args.prompt:
            break


if __name__ == '__main__':
    main()

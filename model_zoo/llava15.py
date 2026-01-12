import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor
from .llava import LlavaForConditionalGeneration, LlavaForConditionalGenerationScal

import torch
import torch.nn.functional as F
import json
import os
import re
import warnings
from typing import List, Optional, Union

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
import transformers
from transformers.generation.utils import (
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput
)

MODEL = 'llava-hf/llava-1.5-7b-hf'


def _add_weight_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    weight: Optional[float] = None,
    adjust_method: Optional[str] = None,
    pos: Optional[torch.Tensor] = None,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    before = () if (return_dict_in_generate) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
    
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
        if 'Scal' not in str(type(self)):
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            outputs = self(
                **model_inputs,
                weight=weight,
                adjust_method=adjust_method,
                pos=pos,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def change_greedy_to_add_weight():
    transformers.generation.utils.GenerationMixin._greedy_search = _add_weight_greedy_search


class LlavaWrapper:
    def __init__(self, root_dir, device, method):
        # Document 5 (llava16.py) 원본 방식 + Document 4 추가 메서드
        method_need_scal = [
            'scaling_vis', 
            'adapt_vis', 
            'adapt_vis_jsd', 
            'adapt_vis_obj',
            'adapt_vis_entropy',  # get_answer()에서 weight 사용
            'adapt_vis_for_oracle_research'  # get_answer()에서 여러 weight 사용
        ]
        
        if method in method_need_scal:
            self.model = LlavaForConditionalGenerationScal.from_pretrained(
                MODEL, revision='a272c74', cache_dir=root_dir,
                ignore_mismatched_sizes=True
            ).eval().to(device)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                MODEL, revision='a272c74', cache_dir=root_dir,
                ignore_mismatched_sizes=True
            ).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            MODEL, revision='a272c74', cache_dir=root_dir
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            MODEL, revision='a272c74', cache_dir=root_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL, revision='a272c74', cache_dir=root_dir
        )
        self.device = device

        # Document 5 원본
        self.relation_option_sets = {
            "Controlled_Images_A": ["Left", "Right", "On", "Under"],
            "Controlled_Images_B": ["Left", "Right", "Front", "Behind"],
            "VG_QA_two_obj": ["left", "right", "front", "behind", "above", "below"],
            "VG_QA_one_obj": ["left", "right", "front", "behind", "above", "below"],
            "VSR": ["Yes", "No"],
        }
        self._relation_token_cache = {}

    # ============================================================================
    # Helper Methods (Document 4 & 5)
    # ============================================================================
    
    def _kl(self, p, q):
        return torch.sum(p * torch.log(p / q))

    def _jsd(self, p, q):
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * torch.log(p / m))
        kl_qm = torch.sum(q * torch.log(q / m))
        return 0.5 * (kl_pm + kl_qm)
    
    def get_distribution(self, score, dataset="Controlled_Images_A"):
        """Document 4"""
        if dataset == "Controlled_Images_A":
            options = ["Left", "Right", "On", "Under"]
        elif dataset == "Controlled_Images_B":
            options = ["Left", "Right", "Front", "Behind"]
        elif dataset == "COCO_QA_one_obj":
            options = ["Left", "Right", "Top", "Bottom"]
        elif dataset == "COCO_QA_two_obj":
            options = ["Left", "Right", "Above", "Below"]
        elif dataset == "VG_QA_one_obj":
            options = ["Left", "Right", "Front", "Behind", "Top", "Bottom"]
        elif dataset == "VG_QA_two_obj":
            options = ["Left", "Right", "Front", "Behind", "Above", "Below"]

        option_ids = [self.tokenizer.encode(r, add_special_tokens=False)[0] for r in options]
        prob_options = score[0, option_ids]
        p = F.softmax(prob_options, dim=-1)
        p = p + 1e-12
        p = p / p.sum()
        
        N = len(options)
        return {options[i]: float(p[i].item()) for i in range(N)}
        
    def get_uncertainty(self, score, distribution, method='entropy'):
        """Document 4"""
        if method == 'confidence':
            res = float(max(torch.nn.functional.softmax(score, dim=-1)[0]))
        else:
            N = len(distribution)
            p = torch.Tensor([distribution[key] for key in distribution]).to(score.device)

            if method == 'kld':
                uniform_dist = torch.ones_like(p) / N
                res = float(self._kl(p, uniform_dist).item())
            elif method == 'jsd':
                uniform_dist = torch.ones_like(p) / N
                res = float(self._jsd(p, uniform_dist).item())
            elif method == 'entropy':
                entropy = -torch.sum(p * torch.log(p))
                max_entropy = torch.log(torch.tensor(float(N)))
                normalized_entropy = entropy / max_entropy
                res = float(normalized_entropy.item())
            else:
                res = float(max(torch.nn.functional.softmax(score, dim=-1)[0]))
        
        return res

    def _extract_entities_from_prompt(self, prompt: str):
        """Document 5"""
        try:
            m = re.search(
                r"Where is the (.+?) in relation to the (.+?)\?",
                prompt,
                re.IGNORECASE,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                return subj, obj
        except Exception:
            pass
        return None, None

    def _build_object_focus_prompt(self, prompt: str):
        """Document 5"""
        subj, obj = self._extract_entities_from_prompt(prompt)
        parts = []
        if subj:
            parts.append(f" the {subj}")
        if obj:
            if subj:
                parts.append(f" and the {obj}")
            else:
                parts.append(f" the {obj}")

        if parts:
            focus_text = " Please look carefully at" + "".join(parts) + "."
            return prompt + focus_text
        else:
            return prompt

    def _get_relation_token_ids(self, dataset: str):
        """Document 5"""
        if dataset not in self.relation_option_sets:
            return None

        if dataset in self._relation_token_cache:
            return self._relation_token_cache[dataset]

        words = self.relation_option_sets[dataset]
        token_ids = []
        for w in words:
            encoded = self.tokenizer(w, add_special_tokens=False)
            if hasattr(encoded, "input_ids"):
                ids = encoded.input_ids
            else:
                ids = encoded["input_ids"]
            if len(ids) == 0:
                continue
            token_ids.append(ids[0])

        if not token_ids:
            return None

        token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        self._relation_token_cache[dataset] = token_ids
        return token_ids

    def _parse_pred_relation(self, generation: str) -> Optional[str]:
        """Document 5"""
        text = generation.lower()
        if re.search(r"\bunder\b", text):
            return "under"
        if re.search(r"\bon\b", text):
            return "on"
        if re.search(r"\bleft\b", text):
            return "left"
        if re.search(r"\bright\b", text):
            return "right"
        return None

    def _compute_jsd_confidence(self, output, dataset: str) -> float:
        """Document 5"""
        first_step_scores = output["scores"][0]
        logits = first_step_scores[0]

        rel_token_ids = self._get_relation_token_ids(dataset)
        if rel_token_ids is None:
            probs_full = torch.softmax(logits, dim=-1)
            conf = probs_full.max().item()
            return float(np.round(conf, 2))

        option_logits = logits[rel_token_ids]
        probs = torch.softmax(option_logits, dim=-1)
        K = probs.size(0)

        uniform = torch.full_like(probs, 1.0 / K)
        m = 0.5 * (probs + uniform)

        jsd = 0.5 * (
            torch.sum(probs * (probs.log() - m.log()))
            + torch.sum(uniform * (uniform.log() - m.log()))
        )

        return float(np.round(jsd.detach().cpu().item(), 2))

    @torch.no_grad()
    def get_answer(self, prompt, image, weight=None, max_length=77, max_new_tokens=100):
        """Document 4 - Helper for reasoning methods
        
        Supports both regular and Scal models:
        - If weight is None or regular model: basic generation
        - If weight is provided and Scal model: weighted generation
        """
        single_input = self.processor(
            text=prompt, images=image, padding="max_length",
            return_tensors="pt", max_length=max_length
        ).to(self.device)
        
        # Check if Scal model and weight provided
        is_scal_model = 'Scal' in str(type(self.model))
        
        if weight is None or not is_scal_model:
            # Basic generation (regular model or no weight)
            output = self.model.generate(
                **single_input, max_new_tokens=max_new_tokens,
                output_scores=True, return_dict_in_generate=True
            )
        else:
            # Weighted generation (Scal model with weight)
            keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
            output = self.model.generate(
                **single_input, keys=keys, weight=weight,
                max_new_tokens=max_new_tokens,
                output_scores=True, return_dict_in_generate=True
            )
        
        gen = self.processor.decode(
            output['sequences'][0][len(single_input['input_ids'][-1]):],
            skip_special_tokens=True
        )
        scores = output['scores']
        
        return gen, scores
    
    # ============================================================================
    # Embedding Methods (Document 5)
    # ============================================================================
    
    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=64, normalize=False):
        num_text = len(texts)
        text_embeds = []
        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i: min(num_text, i+text_batch_size)]
            text_input = self.tokenizer(
                text=text, return_tensors="pt",
                padding="max_length", max_length=77
            ).to(self.device)
            text_feats = self.model.llava.get_text_features(**text_input).cpu().numpy()[:, 0, :].to(self.device)
            if normalize:
                text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)
            text_embeds.append(text_feats)
        return np.concatenate(text_embeds, axis=0)
    
    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        for batch in tqdm(image_loader):
            images = batch["image"]
            inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
            image_feats = self.model.llava.get_image_features(**inputs).cpu().numpy()[:, 0, :]
            if normalize:
                image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
            image_embeds.append(image_feats)
        return np.concatenate(image_embeds, axis=0)
    
    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        text_embeds = self.get_text_embeddings(texts, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        return scores
    
    # ============================================================================
    # Main Evaluation Method
    # ============================================================================
    
    @torch.no_grad()
    def get_out_scores_wh_batched(
        self, dataset, joint_loader, method, weight, option,
        threshold, weight1, weight2
    ):
        scores = []
        index_of_total = 0
        acc = 0
        correct_id = []

        qst_ans_file = f'prompts/{dataset}_with_answer_{option}_options.jsonl'
        
        with open(qst_ans_file, 'r') as file:
            prompt_list = []
            answer_list = []
            for line in file:
                data = json.loads(line)
                prompt_list.append(data["question"])
                answer_list.append(data["answer"])

        SAMPLE = True
        TEST = os.getenv('TEST_MODE', 'False') == 'True'
        total_data_count = len(prompt_list)
        
        if SAMPLE:
            idx_file_path = f'./output/sampled_idx_{dataset}.npy'
            
            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(
                    range(total_data_count),
                    int(0.2 * total_data_count)
                )
                sampled_indices.sort()
                np.save(idx_file_path, np.array(sampled_indices))

            if TEST:
                all_indices = set(range(total_data_count))
                unsampled_indices = list(all_indices - set(sampled_indices))
                unsampled_indices.sort()
                sampled_indices = unsampled_indices

            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        save_attn_dir = f"./output/{dataset}_weight{weight:.2f}"
        os.makedirs(save_attn_dir, exist_ok=True)

        results = []
        output_result_file_path = None
        
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            for i_option in batch["image_options"]:
                im_scores = []
                
                for _ in i_option:
                    result = None
                    prompt = prompt_list[index_of_total]
                    uncertainty = None
                    gen = None

                    # Preprocess input (common)
                    single_input = self.processor(
                        text=prompt, images=_, padding="max_length",
                        return_tensors="pt", max_length=77
                    ).to(self.device)
                    
                    keys = [
                        torch.where(input_id == 32001, 1, 0)
                        for input_id in single_input['input_ids']
                    ]

                    # ============================================================
                    # Method Dispatch - Document 5 원본 우선
                    # ============================================================
                    
                    if method == 'scaling_vis':
                        # Document 5 원본 그대로
                        change_greedy_to_add_weight()
                        output = self.model.generate(
                            **single_input, keys=keys, weight=weight,
                            max_new_tokens=100, output_scores=True,
                            return_dict_in_generate=True
                        )
                        uncertainty = np.round(
                            float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2
                        )
                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True
                        )
                    
                    elif method == 'adapt_vis':
                        # Document 5 원본 그대로
                        change_greedy_to_add_weight()
                        output = self.model.generate(
                            **single_input, weight=1.0, max_new_tokens=100,
                            output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty = np.round(
                            float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2
                        )
                        print(uncertainty, threshold)

                        if uncertainty < threshold:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight1,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                        else:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight2,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True
                        )
                    
                    elif method == 'adapt_vis_jsd':
                        # Document 5 원본 그대로
                        change_greedy_to_add_weight()
                        output = self.model.generate(
                            **single_input, weight=1.0,
                            max_new_tokens=100, output_scores=True,
                            return_dict_in_generate=True
                        )
                        uncertainty = self._compute_jsd_confidence(output, dataset)
                        print("JSD_conf:", uncertainty, "threshold:", threshold)

                        if uncertainty < threshold:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight1,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                        else:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight2,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True
                        )

                    elif method == 'adapt_vis_obj':
                        # Document 5 원본 그대로
                        change_greedy_to_add_weight()
                        output = self.model.generate(
                            **single_input, weight=1.0,
                            max_new_tokens=100, output_scores=True,
                            return_dict_in_generate=True
                        )
                        probs = torch.nn.functional.softmax(output['scores'][0], dim=-1)[0]
                        uncertainty = np.round(float(max(probs)), 2)
                        print("adapt_vis_obj base_conf:", uncertainty, "threshold:", threshold)

                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True
                        )

                        if uncertainty < threshold:
                            focus_prompt = self._build_object_focus_prompt(prompt)
                            focus_input = self.processor(
                                text=focus_prompt, images=_,
                                padding="max_length", return_tensors="pt",
                                max_length=77
                            ).to(self.device)
                            focus_keys = [
                                torch.where(input_id == 32001, 1, 0)
                                for input_id in focus_input['input_ids']
                            ]
                            output = self.model.generate(
                                **focus_input, keys=focus_keys, weight=weight1,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                            gen = self.processor.decode(
                                output['sequences'][0][len(focus_input['input_ids'][-1]):],
                                skip_special_tokens=True
                            )
                    
                    elif method == 'reasoning_absolute_4directions':
                        # Document 4 - 9-grid with strong prompting and fallback inference
                        def extract_position_9grid(text):
                            """Extract 9-grid position from model output"""
                            text = text.lower()
                            # Check combinations first
                            if ('top' in text or 'upper' in text) and 'left' in text:
                                return 'top-left'
                            elif ('top' in text or 'upper' in text) and 'right' in text:
                                return 'top-right'
                            elif ('top' in text or 'upper' in text) and 'center' in text:
                                return 'top-center'
                            elif ('bottom' in text or 'lower' in text) and 'left' in text:
                                return 'bottom-left'
                            elif ('bottom' in text or 'lower' in text) and 'right' in text:
                                return 'bottom-right'
                            elif ('bottom' in text or 'lower' in text) and 'center' in text:
                                return 'bottom-center'
                            elif ('middle' in text or 'center' in text) and 'left' in text:
                                return 'middle-left'
                            elif ('middle' in text or 'center' in text) and 'right' in text:
                                return 'middle-right'
                            elif 'center' in text or 'middle' in text:
                                return 'center'
                            # Fallback to simple directions
                            elif 'top' in text or 'upper' in text:
                                return 'top-center'
                            elif 'bottom' in text or 'lower' in text:
                                return 'bottom-center'
                            elif 'left' in text:
                                return 'middle-left'
                            elif 'right' in text:
                                return 'middle-right'
                            return 'center'
                        
                        def infer_from_positions(pos1, pos2):
                            """Infer relative position from two 9-grid positions"""
                            # Map positions to coordinates
                            pos_map = {
                                'top-left': (0, 0), 'top-center': (1, 0), 'top-right': (2, 0),
                                'middle-left': (0, 1), 'center': (1, 1), 'middle-right': (2, 1),
                                'bottom-left': (0, 2), 'bottom-center': (1, 2), 'bottom-right': (2, 2),
                            }
                            
                            if pos1 not in pos_map or pos2 not in pos_map:
                                return None, None
                            
                            x1, y1 = pos_map[pos1]
                            x2, y2 = pos_map[pos2]
                            
                            # Calculate differences
                            dx = x1 - x2
                            dy = y1 - y2
                            
                            # Determine primary direction
                            if abs(dy) > abs(dx):  # Vertical difference is larger
                                if dy < 0:
                                    return 'on', 'high'  # obj1 above obj2
                                else:
                                    return 'under', 'high'  # obj1 below obj2
                            elif abs(dx) > abs(dy):  # Horizontal difference is larger
                                if dx < 0:
                                    return 'left', 'high'  # obj1 to the left
                                else:
                                    return 'right', 'high'  # obj1 to the right
                            elif dy < 0:  # Equal but prioritize vertical
                                return 'on', 'medium'
                            elif dy > 0:
                                return 'under', 'medium'
                            elif dx < 0:
                                return 'left', 'medium'
                            elif dx > 0:
                                return 'right', 'medium'
                            else:
                                return None, 'low'  # Same position
                        
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        
                        # Step 1: Ask for detailed position
                        prompt_step1_obj1 = f"<image>\nUSER: In which part of the image {be_verb} the {obj1} located? Describe the specific position such as 'top-left corner', 'bottom-right area', 'center', 'middle-left side', etc.\nASSISTANT:"
                        prompt_step1_obj2 = f"<image>\nUSER: In which part of the image is the {obj2} located? Describe the specific position such as 'top-left corner', 'bottom-right area', 'center', 'middle-left side', etc.\nASSISTANT:"
                        gen_step1_obj1, l = self.get_answer(prompt_step1_obj1, _)
                        gen_step1_obj2, l = self.get_answer(prompt_step1_obj2, _)
                        
                        # Extract 9-grid positions
                        pos1 = extract_position_9grid(gen_step1_obj1)
                        pos2 = extract_position_9grid(gen_step1_obj2)
                        
                        # Infer relative position from spatial reasoning
                        inferred_relation, confidence = infer_from_positions(pos1, pos2)
                        
                        # Build reasoning prompt based on inference
                        if inferred_relation and confidence in ['high', 'medium']:
                            # Use strong inference as guidance
                            prompt_final = f"<image>\nUSER: I observe that the {obj1} is in the {pos1.replace('-', ' ')} area and the {obj2} is in the {pos2.replace('-', ' ')} area of the image. Based on their spatial positions, where {be_verb} the {obj1} in relation to the {obj2}? Answer with left, right, on, or under.\nASSISTANT:"
                        else:
                            # Provide positions and let model decide
                            prompt_final = f"<image>\nUSER: The {obj1} is located in the {pos1.replace('-', ' ')} area and the {obj2} is in the {pos2.replace('-', ' ')} area. Where {be_verb} the {obj1} in relation to the {obj2}? Answer with left, right, on or under.\nASSISTANT:"
                        
                        gen, score = self.get_answer(prompt_final, _)
                        score = score[0]
                        uncertainty = np.round(
                            float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2
                        )
                        
                        consistent = (pos1 != pos2)  # Different positions = consistent

                        result = {
                            "Prompt": prompt,
                            "Step1_1_to_2": gen_step1_obj1,
                            "Step1_2_to_1": gen_step1_obj2,
                            "Position_1": pos1,
                            "Position_2": pos2,
                            "Inferred_relation": inferred_relation,
                            "Inference_confidence": confidence,
                            "Is_consistent": consistent,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "uncertainty": uncertainty
                        }
                    
                    elif method == 'reasoning_relative_location':
                        # Document 4 - Always use first answer (no fallback to base)
                        def extract_direction(text):
                            """Extract direction keyword from model output"""
                            text = text.lower()
                            # Check for on/under first (more specific)
                            if 'under' in text or 'below' in text or 'beneath' in text:
                                return 'under'
                            elif 'on' in text and 'front' not in text:  # Avoid "on the front"
                                return 'on'
                            elif 'left' in text:
                                return 'left'
                            elif 'right' in text:
                                return 'right'
                            return None
                        
                        def consistency_check(gen1, gen2):
                            valid_opposite = {
                                'left': 'right', 'right': 'left',
                                'on': 'under', 'under': 'on',
                            }
                            dir1 = extract_direction(gen1)
                            dir2 = extract_direction(gen2)
                            
                            if dir1 is None or dir2 is None:
                                return False
                            
                            return dir1 in valid_opposite and dir2 == valid_opposite[dir1]
                        
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        
                        prompt_step1_obj1 = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Answer with left, right, on or under.\nASSISTANT:"
                        prompt_step1_obj2 = f"<image>\nUSER: Where is the {obj2} in relation to the {obj1}? Answer with left, right, on or under.\nASSISTANT:"
                        gen_step1_obj1, score_step1_obj1 = self.get_answer(prompt_step1_obj1, _)
                        score_step1_obj1 = score_step1_obj1[0]
                        gen_step1_obj2, score_step1_obj2 = self.get_answer(prompt_step1_obj2, _)
                        
                        consistent = consistency_check(gen_step1_obj1, gen_step1_obj2)
                        
                        # Always use first answer (no fallback to base)
                        gen, score = gen_step1_obj1, score_step1_obj1
                        uncertainty = np.round(
                            float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2
                        )
                        
                        result = {
                            "Prompt": prompt,
                            "Step1_1_to_2": gen_step1_obj1,
                            "Step1_2_to_1": gen_step1_obj2,
                            "Is_consistent": consistent,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "uncertainty": uncertainty
                        }
                    
                    elif method == 'reasoning_relative_relationship':
                        # Document 4
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        
                        prompt_step1 = f"<image>\nUSER: Which of the following positional relationships do the {obj1} and the {obj2} have? 1. A left-right relationship in which one object is next to another or 2. an on-under relationship in which one object is placed on or under another object.\nASSISTANT:"
                        gen_step1, score_step1 = self.get_answer(prompt_step1, _)
                        
                        if 'left' in gen_step1 or 'right' in gen_step1 or '1' in gen_step1:
                            prompt_step2 = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Answer with left or right.\nASSISTANT:"
                            gen, score = self.get_answer(prompt_step2, _)
                            score = score[0]
                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2
                            )
                        elif 'on' in gen_step1 or 'under' in gen_step1 or '2' in gen_step1:
                            prompt_step2 = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Answer with on or under.\nASSISTANT:"
                            gen, score = self.get_answer(prompt_step2, _)
                            score = score[0]
                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2
                            )
                        else:
                            gen, score = self.get_answer(prompt, _)
                            score = score[0]
                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2
                            )
                        
                        result = {
                            "Prompt": prompt,
                            "Step1": gen_step1,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "uncertainty": uncertainty
                        }

                    elif method == 'chain_of_thought':
                        # Document 4
                        few_shot_prompt = '''\
USER: Where is the violin in relation to the sofa? Think step by step, then answer about the relation between violin and sofa with left, right, on or under.
ASSISTANT: In this picture, the brown violin is lying on the floor in the bottom-center area, while the beige fabric sofa stands prominently across the center-center region.\
Since the violin is positioned on the floor beneath the raised seat of the sofa without making contact with the cushion, they are separated.\
Because the violin is located vertically lower than the sofa's seat within its footprint, the relative spatial relationship corresponds to the on-under category.\
So, as the final answer to the question of where the violin is in relation to the sofa, the violin is under the sofa.
USER: Where is the calculator in relation to the desk? Think step by step, then answer about the relation between calculator and desk with left, right, on or under.
ASSISTANT: In this picture, the gray calculator is resting on the surface in the center-right section, and the wooden desk occupies the bottom-center to center-center area.\
Because the device is physically supported by the desk's surface with no gap in between, they are touching.\
Since the calculator is placed upon the upper surface of the desk, the relative spatial relationship corresponds to the on-under category.\
So, as the final answer to the question of where the calculator is in relation to the desk, the calculator is on the desk.
USER: Where is the cat in relation to the rug? Think step by step, then answer about the relation between cat and rug with left, right, on or under.
ASSISTANT: In this picture, the white cat is sitting upright in the bottom-right corner, while the patterned rug lies flat covering the bottom-center area.\
Since the cat is sitting on the floor adjacent to the rug's edge rather than on the fabric itself, they are separated.\
Because the cat is positioned to the eastern side of the frame relative to the rug's location, the relative spatial relationship corresponds to the left-right category.\
So, as the final answer to the question of where the cat is in relation to the rug, the cat is right of the rug.
USER: Where is the stapler in relation to the printer? Think step by step, then answer about the relation between stapler and printer with left, right, on or under.
ASSISTANT: In this picture, the blue stapler is sitting on the desk in the center-left region, while the large laser printer sits heavily in the center-right region.\
Since there is a clear span of empty desk surface between the two office supplies, they are separated.\
Because the stapler is positioned on the western side of the frame relative to the printer, the relative spatial relationship corresponds to the left-right category.\
So, as the final answer to the question of where the stapler is in relation to the printer, the stapler is left of the printer.\
'''
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        new_prompt = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Think step by step, then answer about the relation between the {obj1} and the {obj2} with left, right, on or under.\nASSISTANT:"
                        full_prompt = few_shot_prompt + new_prompt
                        
                        generation, score = self.get_answer(
                            full_prompt, _, max_length=1024, max_new_tokens=512
                        )
                        answer = generation.split('.')[-2].strip()
                        answer = answer.split(',')[-1].strip()
                        print(f"Prompt:\n{new_prompt}\nGeneration: {answer}\nGolden: {answer_list[index_of_total][0]}")
                        gen = answer
                        uncertainty = None  # CoT doesn't use uncertainty

                    elif method == 'adapt_vis_entropy':
                        # Document 4
                        change_greedy_to_add_weight()
                        gen, score = self.get_answer(prompt, _, 1.0)
                        score = score[0]
                        distribution_map = self.get_distribution(score, dataset=dataset)
                        uncertainty = self.get_uncertainty(score, distribution_map, 'entropy')
                        
                        if uncertainty > threshold:
                            gen, score = self.get_answer(prompt, _, weight1)
                        else:
                            gen, score = self.get_answer(prompt, _, weight2)

                    elif method == 'adapt_vis_for_oracle_research':
                        # Document 4
                        change_greedy_to_add_weight()
                        original_generation, original_score = self.get_answer(prompt, _, 1.0)
                        original_score = original_score[0]
                        distribution_map = self.get_distribution(original_score, dataset=dataset)
                        uncertainty = np.round(
                            float(max(torch.nn.functional.softmax(original_score, dim=-1)[0])), 2
                        )
                        uncertainty_confidence = self.get_uncertainty(
                            original_score, distribution_map, method='confidence'
                        )
                        uncertainty_kl = self.get_uncertainty(
                            original_score, distribution_map, method='kld'
                        )
                        uncertainty_js = self.get_uncertainty(
                            original_score, distribution_map, method='jsd'
                        )
                        uncertainty_entropy = self.get_uncertainty(
                            original_score, distribution_map, method='entropy'
                        )
                        uncertainties = {
                            "confidence": uncertainty_confidence,
                            "kld": uncertainty_kl,
                            "jsd": uncertainty_js,
                            "entropy": uncertainty_entropy
                        }
                        
                        weights = [0.5, 0.8, 1.2, 1.5, 2.0]
                        gen_map = {
                            1.0: {
                                "Generation": original_generation,
                                "Distribution": distribution_map
                            }
                        }
                        for w in weights:
                            w_generation, w_score = self.get_answer(prompt, _, w)
                            w_score = w_score[0]
                            distribution_map = self.get_distribution(w_score, dataset=dataset)
                            gen_map[w] = {
                                "Generation": w_generation,
                                "Distribution": distribution_map
                            }
                        
                        gen1, gen2 = gen_map[weight1]['Generation'], gen_map[weight2]['Generation']
                        gen = gen1 if uncertainty < threshold else gen2
                        output_result_file_path = f'./output/results_{dataset}_{method}_{weight}_{weight1}_{weight2}_{threshold}_{TEST}.json'

                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Generation_map": gen_map,
                            "Golden": answer_list[index_of_total][0],
                            "Uncertainty": uncertainty,
                            "Uncertainties": uncertainties
                        }
                    
                    else:
                        # Default (base method)
                        output = self.model.generate(
                            **single_input, max_new_tokens=100,
                            output_scores=True, return_dict_in_generate=True
                        )
                        gen = self.processor.decode(
                            output['sequences'][0][len(single_input['input_ids'][-1]):],
                            skip_special_tokens=True
                        )
                        uncertainty = np.round(float(max(output['scores'][0][0])), 2)
                    
                    # ============================================================
                    # Result Collection - Document 5 형식
                    # ============================================================
                    
                    if result is None:
                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "Uncertainty": float(uncertainty) if uncertainty is not None else None,
                        }
                    
                    results.append(result)
                    
                    print(f"Prompt: {prompt}\nGeneration: {gen}\nGolden: {answer_list[index_of_total][0]}")
                    
                    # Check correctness
                    c_option = batch["caption_options"]
                    if 'CoT' in method or method == 'chain_of_thought':
                        if len(list(c_option)) == 4:
                            gen_lower = gen.lower()
                            if answer_list[index_of_total][0].lower() in gen_lower:
                                acc += 1
                                correct_id.append(index_of_total)
                                answers = [1, 0, 0, 0]
                            else:
                                answers = [0, 0, 1, 0]
                    else:
                        if len(list(c_option)) == 4:
                            if (answer_list[index_of_total][0] in gen or 
                                answer_list[index_of_total][0].lower() in gen.lower()) \
                                    and not (answer_list[index_of_total][0].lower() == 'on' and 
                                           'front' in gen.strip().lower()):
                                acc += 1
                                correct_id.append(index_of_total)
                                answers = [1, 0, 0, 0]
                            else:
                                answers = [0, 0, 1, 0]
                        
                        elif len(list(c_option)) == 2:
                            if (answer_list[index_of_total][0] in gen or 
                                answer_list[index_of_total][0].lower() in gen.lower()) \
                                    and not (answer_list[index_of_total][0].lower() == 'on' and 
                                           'front' in gen.strip().lower()):
                                acc += 1
                                correct_id.append(index_of_total)
                                answers = [1, 0]
                            else:
                                answers = [0, 1]

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    index_of_total += 1

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            # Save results - Document 5 형식 (results1.5_)
            if output_result_file_path is None:
                output_result_file_path = f'./output/results1.5_{dataset}_{method}_{weight}_{option}option_{TEST}.json'
            
            print("Saving results to", output_result_file_path)
            with open(output_result_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)
            print(acc, index_of_total, acc / index_of_total)

        # Save final results
        print(acc / index_of_total)
        output_score_file = output_result_file_path.replace(".json", "scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({
                "acc": acc / index_of_total,
                "correct_id": correct_id
            }, fout, ensure_ascii=False, indent=4)

        all_scores = np.concatenate(scores, axis=0)
        if dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
            return (all_scores, [])
        else:
            return (acc / index_of_total, correct_id)
    
    # ============================================================================
    # VSR Evaluation Method (Document 5)
    # ============================================================================
    
    @torch.no_grad()
    def get_judge_scores_vsr_batched(
        self, dataset, joint_loader, method, weight, threshold, weight1, weight2
    ):
        index = 0
        TP, TN, FP, FN = 0, 0, 0, 0

        save_attn_dir = f"./output/{dataset}_weight{weight:.2f}"
        os.makedirs(save_attn_dir, exist_ok=True)
        
        index_of_total = 0
        results = []

        for batch in tqdm(joint_loader):
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            for i_option in batch["image_options"]:
                for c_option in batch["caption_options"]:
                    prompt = "User: <image>\n Determine whether the description about the spatial relationship is correct or not. Answer with yes or no: "
                    qst = [prompt] * len(list(c_option))
                    end_fix = [" Assistant:"] * len(list(c_option))
                    concatenated_list = [s1 + s2 + s3 for s1, s2, s3 in zip(qst, c_option, end_fix)]
                    
                    for idx, text in enumerate(concatenated_list):
                        single_input = self.processor(
                            text=text, images=list(i_option)[idx],
                            padding="max_length", return_tensors="pt", max_length=77
                        ).to(self.device)
                        keys = [
                            torch.where(input_id == 32001, 1, 0)
                            for input_id in single_input['input_ids']
                        ]
                        
                        if method == 'scaling_vis':
                            change_greedy_to_add_weight()
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2
                            )
                            gen = self.processor.decode(
                                output[0][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True, output_attentions=True
                            )
                        
                        elif method == 'adapt_vis':
                            change_greedy_to_add_weight()
                            output = self.model.generate(
                                **single_input, weight=1.0, max_new_tokens=100,
                                output_scores=True, return_dict_in_generate=True
                            )
                            gen = self.processor.decode(
                                output['sequences'][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True, output_attentions=True
                            )
                            uncertainty = np.round(float(max(output['scores'][0][0])), 2)
                            
                            if uncertainty < threshold:
                                output = self.model.generate(
                                    **single_input, keys=keys, weight=weight1,
                                    max_new_tokens=100, output_scores=True,
                                    return_dict_in_generate=True
                                )
                            else:
                                output = self.model.generate(
                                    **single_input, keys=keys, weight=weight2,
                                    max_new_tokens=100, output_scores=True,
                                    return_dict_in_generate=True
                                )
                            gen = self.processor.decode(
                                output[0][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True, output_attentions=True
                            )
                        
                        elif method == 'adapt_vis_jsd':
                            change_greedy_to_add_weight()
                            output = self.model.generate(
                                **single_input, weight=1.0,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                            gen = self.processor.decode(
                                output['sequences'][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True, output_attentions=True
                            )
                            uncertainty = self._compute_jsd_confidence(output, dataset)
                            print("JSD_conf:", uncertainty, "threshold:", threshold)

                            if uncertainty < threshold:
                                output = self.model.generate(
                                    **single_input, keys=keys, weight=weight1,
                                    max_new_tokens=100, output_scores=True,
                                    return_dict_in_generate=True
                                )
                            else:
                                output = self.model.generate(
                                    **single_input, keys=keys, weight=weight2,
                                    max_new_tokens=100, output_scores=True,
                                    return_dict_in_generate=True
                                )
                            gen = self.processor.decode(
                                output[0][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True, output_attentions=True
                            )
                        
                        else:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight,
                                max_new_tokens=100, output_scores=True,
                                return_dict_in_generate=True
                            )
                            uncertainty = np.round(
                                float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2
                            )
                            gen = self.processor.decode(
                                output[0][0][len(single_input['input_ids'][-1]):],
                                skip_special_tokens=True, output_attentions=True
                            )
                        
                        label = int(batch['labels'][0][idx])
                        if label == 1:
                            TP += 1 if 'Yes' in gen else 0
                            FN += 1 if 'Yes' not in gen else 0
                        else:
                            TN += 1 if 'No' in gen else 0
                            FP += 1 if 'No' not in gen else 0
                        
                        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                        
                        gold = 'Yes' if label == 1 else 'No'
                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": gold,
                            "Uncertainty": uncertainty,
                        }
                        results.append(result)
                        index_of_total += 1
                        
                index += 1

        precision = TP / (TP + FN) if (TP + FN) > 0 else 0
        recall = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TN + TP) / (TN + TP + FN + FP) if (TN + TP + FN + FP) > 0 else 0

        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
              f"Accuracy: {accuracy}\n"
              f"Precision: {precision}\n"
              f"Recall: {recall}\n"
              f"F1 Score: {f1_score}")
        
        all_scores = (TP, TN, FP, FN)
        
        output_file_path = f'./output/results_{dataset}_{method}_{weight}.json'
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
        
        output_score_file = output_file_path.replace(".json", "_scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({
                "acc": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1_score
            }, fout, ensure_ascii=False, indent=4)
        
        return all_scores

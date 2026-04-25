#!/usr/bin/env python3
"""
Unified evaluation script for SAT benchmarks across multiple VLM architectures.
Supports: Qwen3-VL, InternVL3, InternVL3.5, Molmo2, Gemma-3, LLaVA-OneVision, MiniCPM-V, SAIL-VL2
Benchmarks: CVBench, BLINK, SAT-v2
Metrics: QAAccuracy (exact match after normalization)

Usage:
    python evaluate.py --model_path /path/to/model --model_name mymodel --datasets cvbench blink sat --batch_size 8
"""

import os
import sys
import re
import json
import argparse
import traceback
import math
from collections import defaultdict
from typing import List, Dict, Any
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    Qwen3VLForConditionalGeneration,
    GenerationConfig,
    Gemma3ForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    AutoModelForImageTextToText,
    InternVLForConditionalGeneration,
)

# -----------------------------------------------------------------------------
# InternVL dynamic image preprocessing helpers
# -----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_area = ratio[0] * ratio[1] * image_size * image_size
        scaled_w = image_size * ratio[0]
        scaled_h = image_size * ratio[1]
        ratio_diff = abs(aspect_ratio - scaled_w / scaled_h)
        if abs(target_area - area) < best_ratio_diff:
            best_ratio_diff = abs(target_area - area)
            best_ratio = ratio
        elif abs(target_area - area) == best_ratio_diff and ratio_diff < abs(
                aspect_ratio - image_size * best_ratio[0] / (image_size * best_ratio[1])):
            best_ratio = ratio
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def internvl_preprocess(images, image_size=448, max_num=12, use_thumbnail=True):
    transform = build_transform(image_size)
    all_pixel_values = []
    num_patches_list = []
    for img in images:
        patches = dynamic_preprocess(img, image_size=image_size, use_thumbnail=use_thumbnail, max_num=max_num)
        pixel_values = torch.stack([transform(p) for p in patches])
        all_pixel_values.append(pixel_values)
        num_patches_list.append(len(patches))
    pixel_values = torch.cat(all_pixel_values, dim=0)
    return pixel_values, num_patches_list


# -----------------------------------------------------------------------------
# LLaVA-OneVision 1.5 workaround (custom model files)
# -----------------------------------------------------------------------------

def _setup_llava_ov_package():
    """Copy custom model files to a temp package so relative imports work."""
    pkg_dir = "/tmp/llavaonevision1_5_pkg"
    if pkg_dir not in sys.path:
        sys.path.insert(0, "/tmp")
    if os.path.exists(pkg_dir):
        return
    llava_path = None
    # Find any LLaVA-OneVision snapshot
    for root, dirs, files in os.walk("/root/autodl-fs/models"):
        if "modeling_llavaonevision1_5.py" in files:
            llava_path = root
            break
    if llava_path is None:
        return
    os.makedirs(pkg_dir, exist_ok=True)
    for f in ["configuration_llavaonevision1_5.py", "modeling_llavaonevision1_5.py",
              "configuration_intern_vit.py", "modeling_intern_vit.py",
              "configuration_rice.py", "modeling_rice.py"]:
        src = os.path.join(llava_path, f)
        if os.path.exists(src):
            os.system(f"cp {src} {pkg_dir}/")
    open(os.path.join(pkg_dir, "__init__.py"), "w").close()
    sys.path.insert(0, "/tmp")


# -----------------------------------------------------------------------------
# Image resizing helper (prevent OOM / hangs on very large images)
# -----------------------------------------------------------------------------

def maybe_resize_image(image, max_size=2048):
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return image


# -----------------------------------------------------------------------------
# Dataset wrappers
# -----------------------------------------------------------------------------

class CVBenchDataset(Dataset):
    def __init__(self, num_data_points: int = None):
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
        self.data = ds.shuffle(seed=42)
        if num_data_points:
            self.data = self.data.select(range(min(num_data_points, len(self.data))))
        self.choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        choices = ex['choices']
        answer_letter = ex['answer'].replace("(", "").replace(")", "")
        answer = choices[self.choice_map[answer_letter]]
        return {
            "images": [maybe_resize_image(ex['image'])],
            "question": ex['question'],
            "choices": choices,
            "answer": answer,
            "dataset_name": f"cvbench_{ex['type']}_{ex['task']}",
        }


class BLINKDataset(Dataset):
    SUBTASKS = ['Multi-view_Reasoning', 'Relative_Depth', 'Spatial_Relation']

    def __init__(self, num_data_points: int = None):
        self.data = []
        per_task = (num_data_points // len(self.SUBTASKS)) if num_data_points else None
        for subtask in self.SUBTASKS:
            ds = load_dataset("BLINK-Benchmark/BLINK", subtask, split="val")
            if per_task:
                ds = ds.select(range(min(per_task, len(ds))))
            for ex in ds:
                self.data.append((ex, subtask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex, subtask = self.data[idx]
        choices = ex['choices']
        answer_letter = ex['answer'].replace("(", "").replace(")", "")
        answer = choices[ord(answer_letter) - ord('A')]
        images = [maybe_resize_image(ex['image_1'])]
        if ex.get('image_2') is not None:
            images.append(maybe_resize_image(ex['image_2']))
        if ex.get('image_3') is not None:
            images.append(maybe_resize_image(ex['image_3']))
        if ex.get('image_4') is not None:
            images.append(maybe_resize_image(ex['image_4']))
        return {
            "images": images,
            "question": ex['prompt'].split("?")[0] + "?",
            "choices": choices,
            "answer": answer,
            "dataset_name": f"BLINK_{subtask}",
        }


class SATDataset(Dataset):
    def __init__(self, num_data_points: int = None):
        ds = load_dataset("array/SAT-v2", split="test", streaming=True)
        self.data = []
        for i, ex in enumerate(ds):
            if num_data_points and i >= num_data_points:
                break
            self.data.append(ex)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "images": [maybe_resize_image(im) for im in ex['images']],
            "question": ex['question'],
            "choices": ex['answers'],
            "answer": ex['correct_answer'],
            "dataset_name": f"SAT_{ex['question_type']}",
        }


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------

def build_prompt(question: str, choices: List[str] = None) -> str:
    if choices and len(choices) > 1:
        choice_str = ", ".join(choices[:-1]) + ", or " + choices[-1]
        return (
            f"Answer the following question with a single word or phrase. "
            f"Question: {question} Choose between the following options: {choice_str}."
        )
    return f"Answer the following question with a single word or phrase. Question: {question}"


# -----------------------------------------------------------------------------
# Model adapters
# -----------------------------------------------------------------------------

class ModelAdapter:
    def __init__(self, model_path: str, device_map: str = "auto", max_new_tokens: int = 128):
        self.model_path = model_path
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    def generate_batch(self, items: List[Dict]) -> List[str]:
        raise NotImplementedError


class QwenVLAdapter(ModelAdapter):
    def load(self):
        from qwen_vl_utils import process_vision_info
        self._process_vision_info = process_vision_info
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'

    def generate_batch(self, items: List[Dict]) -> List[str]:
        messages_batch = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            content = [{"type": "image", "image": im} for im in item["images"]]
            content.append({"type": "text", "text": prompt_text})
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": content},
            ]
            messages_batch.append(messages)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        image_inputs, video_inputs = self._process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        answers = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [a.strip() for a in answers]


class InternVL3Adapter(ModelAdapter):
    """For InternVLChatModel (InternVL3) and SAIL-VL2 which share the same API."""
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()

    def generate_batch(self, items: List[Dict]) -> List[str]:
        results = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            pixel_values, num_patches_list = internvl_preprocess(item["images"])
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
            generation_config = {
                'max_new_tokens': self.max_new_tokens,
                'do_sample': False,
            }
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=prompt_text,
                generation_config=generation_config,
            )
            results.append(response.strip())
        return results


class InternVL35Adapter(ModelAdapter):
    """For InternVL3.5 (HF native InternVLForConditionalGeneration)."""
    def load(self):
        self.model = InternVLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'

    def generate_batch(self, items: List[Dict]) -> List[str]:
        messages_batch = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            content = [{"type": "image"} for _ in item["images"]]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            messages_batch.append(messages)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        images_lists = [item["images"] for item in items]
        max_images = max(len(imgs) for imgs in images_lists)
        if max_images > 1:
            results = []
            for text, imgs in zip(texts, images_lists):
                inputs = self.processor(images=imgs, text=text, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                generated = outputs[:, inputs["input_ids"].shape[1]:]
                ans = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                results.append(ans)
            return results
        else:
            inputs = self.processor(images=[imgs[0] for imgs in images_lists], text=texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = outputs[:, inputs["input_ids"].shape[1]:]
            answers = self.processor.batch_decode(generated, skip_special_tokens=True)
            return [a.strip() for a in answers]


class MolmoAdapter(ModelAdapter):
    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'

    def generate_batch(self, items: List[Dict]) -> List[str]:
        results = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            image_placeholder = " ".join(["<|image|>"] * len(item["images"]))
            prompt_text = f"{image_placeholder}\n{prompt_text}"
            inputs = self.processor(text=prompt_text, images=item["images"][0] if len(item["images"]) == 1 else item["images"], return_tensors="pt")
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = output[:, inputs['input_ids'].shape[1]:]
            ans = self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            results.append(ans)
        return results


class Gemma3Adapter(ModelAdapter):
    def load(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self.processor.tokenizer.padding_side = 'left'

    def generate_batch(self, items: List[Dict]) -> List[str]:
        messages_batch = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            content = []
            for im in item["images"]:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            messages_batch.append(messages)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        inputs = self.processor(text=texts, images=[item["images"] for item in items], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        answers = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [a.strip() for a in answers]


class LLaVAOVAdapter(ModelAdapter):
    def load(self):
        _setup_llava_ov_package()
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING
        from llavaonevision1_5_pkg.configuration_llavaonevision1_5 import Llavaonevision1_5Config
        from llavaonevision1_5_pkg.modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration

        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(cfg.vision_config, 'model_type'):
            rice_cfg_cls = type(cfg.vision_config)
            CONFIG_MAPPING.register(cfg.vision_config.model_type, rice_cfg_cls)
        CONFIG_MAPPING.register('llavaonevision1_5', Llavaonevision1_5Config)
        MODEL_FOR_VISION_2_SEQ_MAPPING.register(Llavaonevision1_5Config, LLaVAOneVision1_5_ForConditionalGeneration)

        self.model = LLaVAOneVision1_5_ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'

    def generate_batch(self, items: List[Dict]) -> List[str]:
        messages_batch = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            content = []
            for im in item["images"]:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            messages_batch.append(messages)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        inputs = self.processor(text=texts, images=[item["images"] for item in items], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        answers = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [a.strip() for a in answers]


class MiniCPMVAdapter(ModelAdapter):
    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'

    def generate_batch(self, items: List[Dict]) -> List[str]:
        results = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            msgs = [{'role': 'user', 'content': item["images"] + [prompt_text]}]
            res = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, max_new_tokens=self.max_new_tokens, sampling=False)
            results.append(res.strip())
        return results


class GenericAdapter(ModelAdapter):
    """Fallback for unknown but AutoModel-loadable architectures."""
    def load(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                trust_remote_code=True,
            ).eval()
        except Exception:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                trust_remote_code=True,
            ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
        self.tokenizer.padding_side = 'left'

    def generate_batch(self, items: List[Dict]) -> List[str]:
        results = []
        for item in items:
            prompt_text = build_prompt(item["question"], item.get("choices"))
            content = []
            for im in item["images"]:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                inputs = self.processor(images=item["images"], text=prompt, return_tensors="pt")
            except Exception:
                inputs = self.processor(text=prompt, images=item["images"][0] if len(item["images"]) == 1 else item["images"], return_tensors="pt")
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = output[:, inputs["input_ids"].shape[1]:]
            ans = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            results.append(ans)
        return results


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

MODEL_ADAPTERS = {
    "qwen": QwenVLAdapter,
    "internvl3": InternVL3Adapter,
    "internvl35": InternVL35Adapter,
    "molmo": MolmoAdapter,
    "gemma": Gemma3Adapter,
    "llava_ov": LLaVAOVAdapter,
    "minicpm": MiniCPMVAdapter,
    "generic": GenericAdapter,
}


def get_adapter(model_path: str, adapter_type: str = None) -> ModelAdapter:
    if adapter_type and adapter_type in MODEL_ADAPTERS:
        return MODEL_ADAPTERS[adapter_type](model_path)
    path_lower = model_path.lower()
    if "qwen" in path_lower and "vl" in path_lower:
        return QwenVLAdapter(model_path)
    if "internvl3_5" in path_lower or "internvl3.5" in path_lower:
        return InternVL35Adapter(model_path)
    if "internvl" in path_lower:
        return InternVL3Adapter(model_path)
    if "molmo" in path_lower:
        return MolmoAdapter(model_path)
    if "gemma" in path_lower:
        return Gemma3Adapter(model_path)
    if "llava" in path_lower and "onevision" in path_lower:
        return LLaVAOVAdapter(model_path)
    if "minicpm" in path_lower:
        return MiniCPMVAdapter(model_path)
    if "sail" in path_lower and "vl" in path_lower:
        return InternVL3Adapter(model_path)
    return GenericAdapter(model_path)


# -----------------------------------------------------------------------------
# Metric: QAAccuracy from the SAT paper
# -----------------------------------------------------------------------------

def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = ans.replace("(", "").replace(")", "")
    ans = ans.replace(".", "").replace("!", "").replace("?", "")
    ans = ans.strip('"').strip("'")
    return ans


def check_answer(pred: str, gt: str, question: str = "") -> int:
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)

    if gt in pred or pred in gt:
        return 1

    if "is closer" in gt:
        gt_word = gt.split(" is closer")[0].split(" ")[-1].strip()
        pred_word = pred.split(" is closer")[0].split(" ")[-1].strip() if "is closer" in pred else pred
        if gt_word in pred_word or pred_word in gt_word:
            return 1

    if "which object is closer" in question.lower():
        if "is closer" in pred:
            pred_word = pred.split(" is closer")[0].split(" ")[-1].strip()
            if gt in pred_word:
                return 1

    if "is the camera moving" in question.lower():
        if "moving" in pred:
            pred_word = pred.split("moving ")[1].split(" ")[0].strip()
            if pred_word == "clockwise":
                pred_word = "left"
            elif pred_word == "counter-clockwise":
                pred_word = "right"
            if gt in pred_word:
                return 1

    if "considering the relative positions" in question.lower():
        if "is located to" in pred:
            if "right" in pred:
                pred = "right"
            else:
                pred = "left"
        if gt in pred or pred in gt:
            return 1

    return 0


# -----------------------------------------------------------------------------
# Evaluation runner
# -----------------------------------------------------------------------------

def collate_fn(batch):
    return batch


def run_evaluation(adapter: ModelAdapter, dataset: Dataset, batch_size: int, output_dir: str, model_name: str, dataset_name: str):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    all_preds = []
    all_gts = []
    all_names = []
    all_questions = []

    for batch in tqdm(dataloader, desc=f"Eval {model_name} on {dataset_name}"):
        try:
            preds = adapter.generate_batch(batch)
        except Exception as e:
            print(f"Error in generation batch: {e}")
            traceback.print_exc()
            preds = [""] * len(batch)
        for item, pred in zip(batch, preds):
            all_preds.append(pred)
            all_gts.append(item["answer"])
            all_names.append(item["dataset_name"])
            all_questions.append(item["question"])

    overall_correct = 0
    per_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, gt, name, q in zip(all_preds, all_gts, all_names, all_questions):
        correct = check_answer(pred, gt, q)
        overall_correct += correct
        per_dataset[name]["correct"] += correct
        per_dataset[name]["total"] += 1

    overall_acc = overall_correct / len(all_gts) if all_gts else 0.0
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "overall_accuracy": overall_acc,
        "total_samples": len(all_gts),
        "per_subtask": {
            k: {"accuracy": v["correct"] / v["total"], "samples": v["total"]} for k, v in per_dataset.items()
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "predictions": [
                {"question": q, "pred": p, "gt": g, "dataset": n, "correct": check_answer(p, g, q)}
                for q, p, g, n in zip(all_questions, all_preds, all_gts, all_names)
            ]
        }, f, indent=2)

    print(f"[{model_name} / {dataset_name}] Overall Acc: {overall_acc:.4f}")
    for k, v in metrics["per_subtask"].items():
        print(f"  {k}: {v['accuracy']:.4f} ({v['samples']} samples)")

    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to local HuggingFace model")
    parser.add_argument("--model_name", default=None, help="Short name for the model")
    parser.add_argument("--adapter", default=None, choices=list(MODEL_ADAPTERS.keys()), help="Force adapter type")
    parser.add_argument("--datasets", nargs="+", default=["cvbench", "blink", "sat"], choices=["cvbench", "blink", "sat"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_data_points", type=int, default=None)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    model_name = args.model_name or os.path.basename(os.path.normpath(args.model_path))
    adapter = get_adapter(args.model_path, args.adapter)
    adapter.max_new_tokens = args.max_new_tokens
    adapter.device_map = args.device_map
    print(f"Loading model: {model_name} from {args.model_path}")
    adapter.load()
    print("Model loaded.")

    all_metrics = []
    for ds_name in args.datasets:
        if ds_name == "cvbench":
            dataset = CVBenchDataset(num_data_points=args.num_data_points)
        elif ds_name == "blink":
            dataset = BLINKDataset(num_data_points=args.num_data_points)
        elif ds_name == "sat":
            dataset = SATDataset(num_data_points=args.num_data_points)
        else:
            continue

        metrics = run_evaluation(adapter, dataset, args.batch_size, args.output_dir, model_name, ds_name)
        all_metrics.append(metrics)

    summary_path = os.path.join(args.output_dir, f"{model_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

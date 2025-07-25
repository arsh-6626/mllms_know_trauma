import os
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import argparse
from tqdm import tqdm
import json
from datasets import load_dataset

from llava_methods import *
from blip_methods import *
from qwen2_5_methods import *
from utils import *
from info import *

def vicrop_qa(model_name, method_name, image_path, question, model, processor, short_question):
    """
    Performs visual cropping and question answering using different attention methods.
    
    This function processes an image with a specified model (LLaVA or BLIP) and attention method,
    generates an attention map, crops the image based on the attention, and then performs
    question answering on both the original and cropped images.
    
    Args:
        model_name: String indicating which model to use ("llava" or "blip")
        method_name: String indicating which attention method to use (e.g., "grad_att", "rel_att", "pure_grad")
        image_path: Path to the input image file
        question: The full question to ask about the image
        model: The loaded model instance (LLaVA or BLIP)
        processor: The processor for the corresponding model
        short_question: A shortened version of the question for attention computation (only used in Vstar)
        
    Returns:
        tuple: (original_answer, cropped_answer, bounding_box)
            - original_answer: Model's answer using the full image
            - cropped_answer: Model's answer using the full image and the cropped image
            - bounding_box: The coordinates of the crop (left, top, right, bottom)
    """

    if model_name == "llava":
        bbox_size = 336
    elif model_name == "blip":
        bbox_size = 224
    elif model_name == "qwen2_5":
        bbox_size = 224

    image = Image.open(image_path).convert("RGB")
    model.eval()

    general_question = 'Write a general description of the image.'

    if model_name == "llava":
        
        short_prompt = f"<image>\nUSER: {short_question} Answer the question using a single word or phrase.\nASSISTANT:"
        prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: {general_question} Answer the question using a single word or phrase.\nASSISTANT:"


        inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(ori_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]

        del inputs
        torch.cuda.empty_cache()

        if method_name == 'grad_att':
            att_maps = gradient_attention_llava(image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        
        elif method_name == 'grad_att_high':
            att_maps = high_res(gradient_attention_llava, image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_maps, image.size, bbox_size)

        #------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == 'rel_att':
            att_maps = rel_attention_llava(image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)

        elif method_name == 'rel_att_high':
            att_maps = high_res(rel_attention_llava, image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_maps, image.size, bbox_size)
        
        #------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == 'pure_grad':
            grad = pure_gradient_llava(image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(grad, image.size, bbox_size)

        elif method_name == 'pure_grad_high':
            grads = high_res(pure_gradient_llava, image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(grads, image.size, bbox_size)

        crop_image = image.crop(bbox)

        multi_prompt = f"<image><image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        multi_inputs = processor(multi_prompt, [image, crop_image], return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

        multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
        multi_generation = [i.split('ASSISTANT: ')[1] for i in processor.batch_decode(multi_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0]

        return ori_generation, multi_generation, bbox, att_maps
    
    elif model_name == "blip":

        short_prompt = f"Question: {short_question} Short answer:"
        prompt = f"Question: {question} Short answer:"
        general_prompt = f"Question: {general_question} Short answer:"

        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        ori_generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ori_generation = processor.batch_decode(ori_generate_ids, skip_special_tokens=True)[0]

        del inputs
        torch.cuda.empty_cache()

        if method_name == 'grad_att':
            att_map = gradient_attention_blip(image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        
        elif method_name == 'grad_att_high':
            att_maps = high_res(gradient_attention_blip, image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_maps, image.size, bbox_size)

        #------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == 'rel_att':
            att_map = rel_attention_blip(image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        
        elif method_name == 'rel_att_high':
            att_map = high_res(rel_attention_blip, image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)

        #------------------------------------------------------------------------------------------------------------------------------------
        elif method_name == 'pure_grad':
            grad = pure_gradient_blip(image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(grad, image.size, bbox_size)
        
        elif method_name == 'pure_grad_high':
            grad = high_res(pure_gradient_blip, image, short_prompt, general_prompt, model, processor)
            bbox = bbox_from_att_image_adaptive(grad, image.size, bbox_size)

        crop_image = image.crop(bbox)

        multi_inputs = processor(images=[image, crop_image], text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

        multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
        multi_generation = processor.batch_decode(multi_generate_ids, skip_special_tokens=True)[0]

        return ori_generation, multi_generation, bbox

    elif model_name == "qwen2_5":

        prompt = f'{question} Answer the question using a single word or phrase.'
        general_prompt = f'{general_question} Answer the question using a single word or phrase.'
        att_map = rel_attention_qwen2_5(image, prompt, general_prompt, model, processor)
        bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        crop_image = image.crop(bbox)
        
        image_str = encode_base64(image)
        crop_image_str = encode_base64(crop_image)

        ori_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
        ori_inputs = prepare_qwen2_5_input(ori_messages, processor).to(model.device, torch.bfloat16)
        ori_generate_ids = model.generate(**ori_inputs, max_new_tokens=20, do_sample=False)
        ori_generate_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(ori_inputs.input_ids, ori_generate_ids)]
        ori_generation = processor.batch_decode(ori_generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        multi_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "image", "image": f'data:image;base64,{crop_image_str}', }, {"type": "text", "text": prompt}]}]
        multi_inputs = prepare_qwen2_5_input(multi_messages, processor).to(model.device, torch.bfloat16)
        multi_generate_ids = model.generate(**multi_inputs, max_new_tokens=20, do_sample=False)
        num_img_tokens = sum(multi_inputs.input_ids[0] == 151655)
        multi_generate_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(multi_inputs.input_ids, multi_generate_ids)]
        multi_generation = processor.batch_decode(multi_generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return ori_generation, multi_generation, bbox, num_img_tokens
        

def main(args):
    """
    Main function to run the visual cropping and question answering pipeline.
    
    This function loads the specified model and processor, processes the dataset,
    applies the visual cropping and question answering to each data point,
    and saves the results to a JSON file.
    
    Args:
        args: An argparse.Namespace object containing the following attributes:
            - model: String indicating which model to use ("llava" or "blip")
            - model_id: The model identifier for loading from HuggingFace
            - device: The device to run the model on ("cuda" or "cpu")
            - question_path: Path to the question dataset
            - image_path: Path to the directory containing images
            - task: The task identifier
            - method: The attention method to use
            - output_path: Path to save the results
            - total_chunks: Total number of chunks to split the dataset into
            - chunk_id: The ID of the current chunk to process
            
    Returns:
        None: Results are saved to the specified output file
    """

    if args.model == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_id)
    elif args.model == 'blip':
        model = InstructBlipForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
        processor = InstructBlipProcessor.from_pretrained(args.model_id)
    elif args.model == 'qwen2_5':
        max_pixels = 256 * 28 * 28
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=max_pixels)
        processor.image_processor.size["longest_edge"] = max_pixels # this is likely a bug in current transformers (4.50.0) library, passing in max_pixels to from_pretrained does not work
    
    if os.path.exists(args.question_path):
        with open(args.question_path, "r") as f:
            whole_data = json.load(f)
    else:
        whole_data = list(load_dataset(args.question_path)['test'])
    
    for data in whole_data:
        data["image_path"] = os.path.join(args.image_path, data["image_path"]) if "image_path" in data else os.path.join(args.image_path, f"{data['image_id']}.jpg")

    splited_data = np.array_split(whole_data, args.total_chunks)

    data = splited_data[args.chunk_id]

    new_datas = []

    for d in tqdm(data, desc="Processing", ncols=100):

        question = d["question"]
        image_path = d["image_path"]
        if 'short_question' in d:
            short_question = d["short_question"]
        else:
            short_question = d["question"]

        if args.model == "qwen2_5":
            ori_generation, crop_generation, bbox, num_img_tokens = vicrop_qa(args.model, args.method, image_path, question, model, processor, short_question)
            d["num_img_tokens"] = int(num_img_tokens)
        else:
            ori_generation, crop_generation, bbox = vicrop_qa(args.model, args.method, image_path, question, model, processor, short_question)

        d["original_answer"] = ori_generation
        d["crop_answer"] = crop_generation
        d["bbox"] = bbox

        new_datas.append(d)

    out_put_dir = os.path.dirname(args.output_path)
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            old_datas = json.load(f)
        new_datas = old_datas + new_datas
    
    with open(args.output_path, "w") as f:
        json.dump(new_datas, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava", choices=model_to_fullname.keys())
    parser.add_argument("--task", type=str, default="textvqa", choices=task_to_question_path.keys())
    parser.add_argument("--method", type=str, default="new", choices=["rel_att", "pure_grad", "grad_att", "grad", "rel_att_high", "pure_grad_high", "grad_att_high", "grad_high"])
    parser.add_argument("--save_path", type=str, default="./playground/data/results")
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    output_name = f'{args.model}-{args.task}-{args.method}.json'

    args.model_id = model_to_fullname[args.model]

    args.output_path = os.path.join(args.save_path, output_name)

    args.image_path = task_to_image_path[args.task]

    args.question_path = task_to_question_path[args.task]

    main(args)

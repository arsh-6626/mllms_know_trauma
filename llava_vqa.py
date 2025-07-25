from transformers import pipeline
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image, ImageDraw
from run import vicrop_qa
from utils import *
from llava_methods import *

model = LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to('cuda')
processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")
model_name = 'llava'
method_name = 'rel_att'
image_path = 'images/demo1.png'
output_path = "./"
question = 'Find wounds showing red blood in the given image'
short_question = 'Find wounds showing red blood in the given image'
general_question = 'Write a general description of the image.'
short_prompt = f"<image>\nUSER: {short_question} Answer the question using a single word or phrase.\nASSISTANT:"
prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
general_prompt = f"<image>\nUSER: {general_question} Answer the question using a single word or phrase.\nASSISTANT:"
image = Image.open(image_path).convert("RGB")
ori_answer, crop_answer, bbox, att_map = vicrop_qa(model_name, method_name, image_path, question, model, processor, short_question)
att_map = high_res(rel_attention_llava(image, short_prompt, general_prompt, model, processor))
overlay_image = create_att_overlay(image, att_map,output_path)
visualize_attention_process(image, att_map)

messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "url": output_path},
          {"type": "text", "text": "Which body part does the highlighted part represents ?"},
        ],
    },
]

output = pipe(text=messages, max_new_tokens=30)
print(output)

import torchvision.transforms.functional as TF
import numpy as np
import os
from scipy.ndimage import median_filter, binary_dilation
from skimage.measure import block_reduce
from io import BytesIO
import base64
import cv2
import matplotlib.pyplot as plt

def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_qwen2_5_input(messages, processor):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs

def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):
    """
    Applies a high-pass filter to an image to highlight edges and fine details.
    
    This function resizes the image, applies a Gaussian blur to create a low-frequency version,
    subtracts it from the original to get high-frequency components, and then applies median filtering.
    
    Args:
        image: Input PIL image
        resolusion: Target resolution to resize the image to
        km: Kernel size for median filtering (default: 7)
        kh: Kernel size for Gaussian blur (default: 3)
        reduce: Whether to reduce the output size using block reduction (default: True)
        
    Returns:
        h_brightness: A 2D numpy array representing the high-frequency components of the image
    """

    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=1024):
    """
    Splits a high-resolution image into smaller patches.
    
    This function divides a large image into smaller patches to process them individually,
    which is useful for handling high-resolution images that might be too large for direct processing.
    
    Args:
        image: Input PIL image
        res_threshold: Maximum resolution threshold before splitting (default: 1024)
        
    Returns:
        tuple: (split_images, vertical_split, horizontal_split)
            - split_images: List of PIL image patches
            - vertical_split: Number of vertical splits
            - horizontal_split: Number of horizontal splits
    """

    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

def high_res(map_func, image, prompt, general_prompt, model, processor):
    """
    Applies an attention mapping function to high-resolution images by splitting and recombining.
    
    This function splits a high-resolution image into smaller patches, applies the specified
    attention mapping function to each patch, and then recombines the results into a single
    attention map.
    
    Args:
        map_func: The attention mapping function to apply to each patch
        image: Input PIL image
        prompt: Text prompt for the attention function
        general_prompt: General text prompt for baseline comparison
        model: Model instance (LLaVA or BLIP)
        processor: Processor for the corresponding model
        
    Returns:
        block_att: A 2D numpy array representing the combined attention map for the entire image
    """

    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image)
    att_maps = []
    for split_image in split_images:
        att_map = map_func(split_image, prompt, general_prompt, model, processor)
        # att_map = att_map / att_map.mean()
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])

    return block_att
def create_att_overlay(image, att_map, output_path="./overlay.png", alpha_base=0.4, top_percentile=97):
    """
    Creates an attention overlay on the input image, highlighting low-attention regions with
    a semi-transparent black overlay while leaving high-attention areas unmasked.
    The overlay is also saved using matplotlib.
    """
    # Convert PIL → RGB array if needed
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))

    img_h, img_w = image.shape[:2]

    # Normalize attention to [0,1]
    att_norm = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-12)

    # Build mask of high-attention patches
    thresh = np.percentile(att_norm, top_percentile)
    high_mask = att_norm > thresh

    # Dilate so we buffer around peaks
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], dtype=bool)
    buffered = binary_dilation(high_mask, structure=kernel)

    # Resize masks to full image resolution
    att_resized_mask = cv2.resize(buffered.astype(np.uint8),
                                  (img_w, img_h),
                                  interpolation=cv2.INTER_NEAREST)

    # Initialize alpha map: everywhere = alpha_base
    alpha_map = np.ones((img_h, img_w), dtype=np.float32) * alpha_base

    # For buffered high-attention regions → no overlay (alpha=0)
    alpha_map[att_resized_mask > 0] = 0.0

    # Create overlay color (black)
    overlay_color = np.array([0, 0, 0], dtype=np.uint8)
    overlay = np.full_like(image, overlay_color)

    # Alpha-blend
    blended = image.astype(np.float32).copy()
    for c in range(3):
        blended[:, :, c] = (1 - alpha_map) * image[:, :, c] + alpha_map * overlay[:, :, c]
    blended = blended.astype(np.uint8)

    # Save using matplotlib to preserve layout
    plt.figure(figsize=(img_w/100, img_h/100), dpi=100)
    plt.imshow(blended)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return blended

def visualize_attention_process(image, att_map, output_dir="./"):
    """
    Creates a comprehensive visualization showing the attention processing steps.
    
    Args:
        image: Input image
        att_map: 2D attention map from gradient_attention_llava function
        output_dir: Directory to save output images
    """
    
    # Convert PIL Image to numpy array if needed
    if hasattr(image, 'convert'):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = image.copy()
    
    # Create the main overlay
    result = create_att_overlay(image_np, att_map, f"{output_dir}/attention_overlay.png")
    
    # Create additional visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Attention map
    im1 = axes[0, 1].imshow(att_map, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title("Attention Map")
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # High attention mask with buffer
    att_normalized = (att_map - att_map.min()) / (att_map.max() - att_map.min())
    attention_threshold = np.percentile(att_normalized, 70)
    high_attention_mask = att_normalized > attention_threshold
    struct_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    buffered_mask = binary_dilation(high_attention_mask, structure=struct_element)
    
    axes[1, 0].imshow(buffered_mask, cmap='gray')
    axes[1, 0].set_title("High Attention Areas + Buffer")
    axes[1, 0].axis('off')
    
    # Final result
    axes[1, 1].imshow(result)
    axes[1, 1].set_title("Final Overlay Result")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attention_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved attention overlay to: {output_dir}/attention_overlay.png")
    print(f"Saved analysis visualization to: {output_dir}/attention_analysis.png")
 

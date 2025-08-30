import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import sys
# sys.path.append("..")
from IPython import embed
from tqdm import tqdm
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator



def segment_individual_tree(anns, depth_image, results_path):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    
    # create a frame template for all trees with different size
    individual_frame = np.zeros((sorted_anns[1]['segmentation'].shape[0], sorted_anns[1]['segmentation'].shape[1], 3))
    
    img[:,:,3] = 0
    for idx, ann in enumerate(tqdm(sorted_anns)):
        
        # visualization on RGB image
        m = ann['segmentation']
        
        color_mask = np.concatenate([np.random.random(3), [0.65]])
        img[m] = color_mask
        
        # segment individual tree for depth map
        # skip the first mask. The first mask is the ground
        if idx != 0:
            
            depth_frame = individual_frame.copy()
            individual_mask = (m.copy() * 255).astype(np.uint8)
            # embed()
            
            # image optimization to remove holds
            kernel = np.ones((13, 13), np.uint8) 
            img_dilation = cv2.dilate(individual_mask, kernel, iterations=1) 
            img_erosion = cv2.erode(img_dilation, kernel, iterations=1) 
            # img_erosion = img_dilation
            
            bbox = np.where(img_erosion != 0)
            y_min, y_max = bbox[0].min(), bbox[0].max()
            x_min, x_max = bbox[1].min(), bbox[1].max()
            
            # crop the individual tree and centralize the image to the frame.
            individual_tree_depth = depth_image * img_erosion[:, :, None].astype(bool)
            individual_tree_depth = individual_tree_depth[y_min:y_max, x_min:x_max]
            x_margin = (depth_frame.shape[1] - (x_max - x_min)) // 2
            y_margin = (depth_frame.shape[0] - (y_max - y_min)) // 2
            
            depth_frame[y_margin: y_margin+individual_tree_depth.shape[0], x_margin:x_margin+individual_tree_depth.shape[1], :] += individual_tree_depth
            # depth = depth_image[]
            # cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(results_path, "%d_depth.jpg"%idx), depth_frame)

            
    ax.imshow(img)

def segment_forest(filepath, depth_filepath, results_path):
    image = cv2.imread(filepath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth_image = cv2.imread(depth_filepath)
    # depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    device = torch.device("cuda")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )

    masks = mask_generator_2.generate(image)

    print(len(masks))
    print(masks[0].keys())


    plt.figure(figsize=(20,20))
    plt.imshow(image)
    segment_individual_tree(masks, depth_image, results_path)
    plt.axis('off')
    plt.savefig('masked.png')

    embed()

if __name__ == "__main__":
    
    np.random.seed(3)
    
    prefix = "footprint"
    filepath = "../data/%s.png"%prefix
    depth_filepath = "../data/%s_depth.jpg"%prefix
    results_path = os.path.join("../results", prefix)
    
    os.system("rm -r %s"%results_path)
    os.makedirs(results_path, exist_ok=True)
    
    segment_forest(filepath, depth_filepath, results_path)
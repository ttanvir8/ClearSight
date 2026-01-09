import os
import cv2
import numpy as np
from utils.metrics import calculate_psnr, calculate_ssim

# Paths
gt_path = '/home/tanvir/Work/754/project/other_code/ClearSight/data/toy_data/val/gt'
results_path = '/home/tanvir/Work/754/project/other_code/ClearSight/results/toy_evaluation/ToyData/val'

if not os.path.exists(results_path):
    print(f"Results path {results_path} does not exist yet. Please wait for evaluation to finish.")
    exit(1)

# Get image names
# Note: Output images might have "_output.png" suffix based on restoration.py
# restore() -> utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_output.png"))
# y is the image ID.
# GT images are in `gt_path`. Their names usually don't have _output.
# We need to match them.

imgsName = sorted(os.listdir(results_path))
# Filter for .png
imgsName = [f for f in imgsName if f.endswith('.png')]

cumulative_psnr, cumulative_ssim = 0, 0
count = 0

print(f"Calculating metrics for {len(imgsName)} images...")

for img_name in imgsName:
    # img_name is like "image_id_output.png"
    # we need "image_id.png" or "image_id.jpg" depending on original format.
    # checking toy_data.py -> filenames invoke .jpg usually.
    # Let's check what the ID actually is. 
    # In restoration.py: print(f"starting processing from image {y}") -> y is strictly the ID (no extension).
    # So `y` comes from `img_id` in `toy_data.py`.
    # img_id = re.split('/', input_name)[-1][:-4]
    # So if input is "input/abc.jpg", img_id is "abc".
    # Output file is "abc_output.png".
    
    if not img_name.endswith('_output.png'):
        continue
        
    base_id = img_name.replace('_output.png', '')
    
    # We need to find the corresponding GT file.
    # In toy_data logic: gt_name = input_name.replace('input', 'gt')
    # But we don't have the map here easily without reading the list again.
    # However, we know they are in `gt_path`.
    # Let's verify extension. It was .jpg in the list.
    
    gt_filename = f"{base_id}.jpg"
    gt_full_path = os.path.join(gt_path, gt_filename)
    
    # Sometimes GT are png? Let's check.
    if not os.path.exists(gt_full_path):
        gt_filename = f"{base_id}.png"
        gt_full_path = os.path.join(gt_path, gt_filename)
        
    if not os.path.exists(gt_full_path):
        # Fallback search
        found = False
        for f in os.listdir(gt_path):
            if f.startswith(base_id) and (f.endswith('.jpg') or f.endswith('.png')):
                gt_full_path = os.path.join(gt_path, f)
                found = True
                break
        if not found:
            print(f"Warning: GT for {base_id} not found.")
            continue

    # Read images
    res = cv2.imread(os.path.join(results_path, img_name), cv2.IMREAD_COLOR)
    gt = cv2.imread(gt_full_path, cv2.IMREAD_COLOR)
    
    # Resize GT to match RES if needed (though restoration usually keeps size, let's correspond to what the model output)
    if res.shape != gt.shape:
        gt = cv2.resize(gt, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)

    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    
    # print('Image: %s | PSNR: %.4f | SSIM: %.4f' % (base_id, cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    count += 1

if count > 0:
    print('Average PSNR: %.4f' % (cumulative_psnr / count))
    print('Average SSIM: %.4f' % (cumulative_ssim / count))
else:
    print("No images processed.")

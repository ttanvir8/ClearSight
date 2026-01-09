import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import math

def calculate_psnr(img1, img2):
    # img1, img2: torch tensors [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def main():
    gt_path = '/home/tanvir/Work/754/project/other_code/ClearSight/data/toy_data/val/gt'
    results_path = '/home/tanvir/Work/754/project/other_code/ClearSight/results/toy_evaluation/ToyData/val'
    
    if not os.path.exists(results_path):
        print(f"Results path {results_path} does not exist.")
        return

    imgsName = sorted(os.listdir(results_path))
    imgsName = [f for f in imgsName if f.endswith('.png')]
    
    cumulative_psnr = 0
    cumulative_ssim = 0
    count = 0
    
    print(f"Calculating metrics for {len(imgsName)} images using PyTorch...")
    
    for img_name in imgsName:
        if not img_name.endswith('_output.png'):
            continue
            
        # Parse weird filename: "['id']_output.png"
        base_id = img_name.replace("_output.png", "")
        base_id = base_id.replace("['", "").replace("']", "").replace("'", "")
        
        # Find GT
        gt_full_path = None
        for ext in ['.jpg', '.png']:
            path = os.path.join(gt_path, f"{base_id}{ext}")
            if os.path.exists(path):
                gt_full_path = path
                break
        
        if gt_full_path is None:
             print(f"Warning: GT for {base_id} not found.")
             continue
             
        # Load images
        res_pil = Image.open(os.path.join(results_path, img_name)).convert('RGB')
        gt_pil = Image.open(gt_full_path).convert('RGB')
        
        # Resize GT if needed
        if res_pil.size != gt_pil.size:
             gt_pil = gt_pil.resize(res_pil.size, Image.BICUBIC)
             
        # Convert to Tensor [0, 1]
        res_tensor = F.to_tensor(res_pil).unsqueeze(0) # B, C, H, W
        gt_tensor = F.to_tensor(gt_pil).unsqueeze(0)
        
        # Calculate
        cur_psnr = calculate_psnr(res_tensor, gt_tensor).item()
        cur_ssim = calculate_ssim(res_tensor, gt_tensor).item()
        
        # print(f"Image: {base_id} | PSNR: {cur_psnr:.4f} | SSIM: {cur_ssim:.4f}")
        
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        count += 1
        
    if count > 0:
        print(f"Average PSNR: {cumulative_psnr / count:.4f}")
        print(f"Average SSIM: {cumulative_ssim / count:.4f}")
    else:
        print("No images processed.")

if __name__ == "__main__":
    main()

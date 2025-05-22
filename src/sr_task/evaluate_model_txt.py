import pytorch_msssim
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import os
import sys
import lpips
import statistics
import cv2

def calculate_mmd(real_image_path, generated_image_path, device):
    real_image_pil = Image.open(real_image_path).convert('RGB')
    generated_image_pil = Image.open(generated_image_path).convert('RGB')
    transform = transforms.ToTensor()
    real_image_tensor = transform(real_image_pil).unsqueeze(0)
    generated_image_tensor = transform(generated_image_pil).unsqueeze(0)
    real_image_tensor = real_image_tensor.to(device)
    generated_image_tensor = generated_image_tensor.to(device)
    xx = torch.matmul(real_image_tensor.view(real_image_tensor.size(0), -1), real_image_tensor.view(real_image_tensor.size(0), -1).t())

    yy = torch.matmul(generated_image_tensor.view(generated_image_tensor.size(0), -1), generated_image_tensor.view(generated_image_tensor.size(0), -1).t())
    xy = torch.matmul(real_image_tensor.view(real_image_tensor.size(0), -1), generated_image_tensor.view(generated_image_tensor.size(0), -1).t())

    beta = 1. / (real_image_tensor.size(1) ** 2)
    gamma = 2. / (real_image_tensor.size(1) ** 2)
    mmd = beta * (torch.sum(xx) + torch.sum(yy)) - gamma * torch.sum(xy)
    return mmd.item()

def calculate_lpips(real_image_path, generated_image_path, loss_fn_alex):
    image0 = Image.open(real_image_path)
    image1 = Image.open(generated_image_path)

    transform = transforms.Compose([transforms.PILToTensor()])

    img0 = transform(image0)
    img1 = transform(image1)

    d = loss_fn_alex(img0, img1)
    return d.item()

def calculate_msssim(real_image_path, generated_image_path, device):
    img1_pil = Image.open(real_image_path).convert('L')
    img2_pil = Image.open(generated_image_path).convert('L')
    img1_np = np.array(img1_pil) / 255.0
    img2_np = np.array(img2_pil) / 255.0
    img1 = torch.unsqueeze(torch.tensor(img1_np, dtype=torch.float32), 0).unsqueeze(0).to(device)
    img2 = torch.unsqueeze(torch.tensor(img2_np, dtype=torch.float32), 0).unsqueeze(0).to(device)
    return pytorch_msssim.msssim(img1, img2).item()

def calculate_metrics(real_image_path, generated_image_path, device, loss_fn_alex):
    msssim_score = calculate_msssim(real_image_path, generated_image_path, device)
    mmd_score = calculate_mmd(real_image_path, generated_image_path, device)
    lpips_score = calculate_lpips(real_image_path, generated_image_path, loss_fn_alex)

    real_image = cv2.imread(real_image_path)
    generated_image = cv2.imread(generated_image_path)
    real_gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(real_gray, generated_gray)
    psnr_score = psnr(real_image, generated_image)

    return ssim_score, psnr_score, mmd_score, msssim_score, lpips_score

if len(sys.argv) != 6:
    print("Usage: python evaluate_model.py <same structure?> <test.txt> gt_base_dir gen_img_dir output_dir")
    sys.exit(1)

flag = sys.argv[1]
ground_truth_dir_file = sys.argv[2]
gt_base_dir = sys.argv[3]
generated_dir = sys.argv[4]
output_directory = sys.argv[5]

output_file_path = os.path.join(output_directory, "full_evaluation.txt")
output_file = open(output_file_path, "w")

ssim_scores = []
psnr_scores = []
mmd_scores = []
msssim_scores = []
lpips_scores = []

with open(ground_truth_dir_file, 'r') as f:
    gt_image_paths = f.read().splitlines()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn_alex = lpips.LPIPS(net='alex')

for gt_image_path in gt_image_paths:
    if flag == "no":
        generated_image_file = os.path.join(generated_dir, os.path.basename(gt_image_path).replace('.jpg', '_out.jpg'))
    else:
        generated_image_file = os.path.join(generated_dir, gt_image_path.replace('.jpg', '_out.jpg'))
        os.makedirs(os.path.dirname(generated_image_file), exist_ok=True)
    if os.path.exists(generated_image_file):
        try:
            ssim_score, psnr_score, mmd_score, msssim_score, lpips_score = calculate_metrics(os.path.join(gt_base_dir, gt_image_path), generated_image_file, device, loss_fn_alex)
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
            mmd_scores.append(mmd_score)
            msssim_scores.append(msssim_score)
            lpips_scores.append(lpips_score)

            output_file.write(f"Image pair: {gt_image_path} -> {os.path.basename(generated_image_file)}\n")

            output_file.write(f"SSIM: {ssim_score}\n")
            output_file.write(f"PSNR: {psnr_score}\n")
            output_file.write(f"MMD: {mmd_score}\n")
            output_file.write(f"SSSIM: {msssim_score}\n")
            output_file.write(f"LPIPS: {lpips_score}\n")
            output_file.write("\n")
        except Exception as e:
            print(f"Error processing image pair {gt_image_path} and {generated_image_file}: {e}")
            output_file.write(f"Error processing image pair {gt_image_path} and {generated_image_file}: {e}\n\n")

average_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
average_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0
average_mmd = sum(mmd_scores) / len(mmd_scores) if mmd_scores else 0
average_msssim = sum(msssim_scores) / len(msssim_scores) if msssim_scores else 0
average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0

quartiles_ssim = np.percentile(ssim_scores, [25, 50, 75]) if ssim_scores else [0, 0, 0]
quartiles_psnr= np.percentile(psnr_scores, [25, 50, 75]) if psnr_scores else [0, 0, 0]
quartiles_mmd = np.percentile(mmd_scores, [25, 50, 75]) if mmd_scores else [0, 0, 0]
quartiles_msssim = np.percentile(msssim_scores, [25, 50, 75]) if msssim_scores else [0, 0, 0]
quartiles_lpips = np.percentile(lpips_scores, [25, 50, 75]) if lpips_scores else [0, 0, 0]

min_ssim = min(ssim_scores) if ssim_scores else 0
min_psnr = min(psnr_scores) if psnr_scores else 0
min_mmd = min(mmd_scores) if mmd_scores else 0
min_msssim = min(msssim_scores) if msssim_scores else 0
min_lpips = min(lpips_scores) if lpips_scores else 0

max_ssim = max(ssim_scores) if ssim_scores else 0
max_psnr = max(psnr_scores) if psnr_scores else 0
max_mmd = max(mmd_scores) if mmd_scores else 0
max_msssim = max(msssim_scores) if msssim_scores else 0
max_lpips = max(lpips_scores) if lpips_scores else 0

std_ssim = statistics.stdev(ssim_scores) if len(ssim_scores) > 1 else 0
std_psnr = statistics.stdev(psnr_scores) if len(psnr_scores) > 1 else 0
std_mmd = statistics.stdev(mmd_scores) if len(mmd_scores) > 1 else 0
std_msssim = statistics.stdev(msssim_scores) if len(msssim_scores) > 1 else 0
std_lpips = statistics.stdev(lpips_scores) if len(lpips_scores) > 1 else 0

output_file.write(generated_dir)
output_file.write("\n\n")

output_file.write("Statistics:\n")

output_file.write("SSIM:\n")
output_file.write(f"Average: {average_ssim}\n")
output_file.write(f"Quartiles: {quartiles_ssim}\n")
output_file.write(f"Min: {min_ssim}\n")
output_file.write(f"Max: {max_ssim}\n")
output_file.write(f"Standard Deviation: {std_ssim}\n\n")

output_file.write("PSNR:\n")
output_file.write(f"Average: {average_psnr}\n")
output_file.write(f"Quartiles: {quartiles_psnr}\n")
output_file.write(f"Min: {min_psnr}\n")
output_file.write(f"Max: {max_psnr}\n")
output_file.write(f"Standard Deviation: {std_psnr}\n\n")

output_file.write("MMD:\n")
output_file.write(f"Average: {average_mmd}\n")
output_file.write(f"Quartiles: {quartiles_mmd}\n")
output_file.write(f"Min: {min_mmd}\n")
output_file.write(f"Max: {max_mmd}\n")
output_file.write(f"Standard Deviation: {std_mmd}\n\n")

output_file.write("SSSIM:\n")
output_file.write(f"Average: {average_msssim}\n")
output_file.write(f"Quartiles: {quartiles_msssim}\n")
output_file.write(f"Min: {min_msssim}\n")
output_file.write(f"Max: {max_msssim}\n")
output_file.write(f"Standard Deviation: {std_msssim}\n\n")

output_file.write("LPIPS:\n")
output_file.write(f"Average: {average_lpips}\n")
output_file.write(f"Quartiles: {quartiles_lpips}\n")
output_file.write(f"Min: {min_lpips}\n")
output_file.write(f"Max: {max_lpips}\n")
output_file.write(f"Standard Deviation: {std_lpips}\n\n")

output_file.close()

print("Metrics calculation completed. Results saved in 'full_evaluation.txt'.")

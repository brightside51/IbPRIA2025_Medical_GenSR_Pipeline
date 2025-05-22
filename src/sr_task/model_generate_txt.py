import os
import cv2
import torch
import sys
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
sys.path.append('Real-ESRGAN')
from realesrgan import RealESRGANer
import sys
from pathlib import Path
from PIL import Image, ImageSequence
from torchvision import transforms, utils


if len(sys.argv) < 5 or len(sys.argv) > 6:
    print("Usage: python test_model.py <path before relative> <input_file> <model_path> <output_directory> [scale]")
    sys.exit(1)

rel_path = sys.argv[1]
input_file = sys.argv[2]
model_path = sys.argv[3]
output_dir = sys.argv[4]
model_name = 'SwinIR'

if len(sys.argv) == 6:
    scale = int(sys.argv[5])
else:
    scale = 4

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None)

with open(input_file, 'r') as f:
    input_paths = f.read().splitlines()
for input_path in input_paths:
    #full_path = os.path.join(rel_path, input_path)
    full_path = Path(f"{rel_path}{input_path}")
    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
        continue
    filename = os.path.basename(input_path)
    print(filename)
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.pt', '.gif')):
        print(f"Unsupported file format: {filename}")
        continue

    if model_name == 'MedDiff':
        output_path = Path(f"{output_dir}{input_path.replace(".pt", "_sr.pt")}")
        #output_path = Path("{a}{b}".format(a = output_dir, b = input_path.replace(".pt", "_sr.pt")))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = torch.load(full_path, map_location = torch.device('cuda')).cpu().numpy()
        #img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        print(img.shape)
        output = torch.empty(img.shape[2], img.shape[3] * 2, img.shape[4] * 2)
        for slice in range(img.shape[2]):
            print(img[0, 0, slice, :, :].shape)
            img_slice = Image.fromarray(img[0, 0, slice, :, :]).resize((64, 64))
            output_slice, _ = upsampler.enhance(np.array(img_slice, dtype=np.float32), outscale = 4)
            #output_slice, _ = upsampler.enhance(torch.rand((64, 64)).numpy(), outscale=scale)
            print(output_slice.shape)
            output[slice, :, :] = torch.Tensor(output_slice)
        print(output.shape)
    
    elif model_name == 'VideoDiff':
        #https://github.com/python-pillow/Pillow/issues/5307
        output_path = Path(f"{output_dir}{input_path.replace(".gif", "_sr.pt")}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.open(full_path)
        img = [frame.convert('L') for frame in ImageSequence.Iterator(img)]
        output = torch.empty(len(img), np.array(img[0]).shape[0] * 4, np.array(img[1]).shape[0] * 4)
        for slice in range(len(img)):
            print(np.array(img[slice], dtype=np.float32).shape)
            output_slice, _ = upsampler.enhance(np.array(img[slice], dtype=np.float32), outscale=scale)
            print(output_slice.shape)
            output[slice, :, :] = torch.Tensor(output_slice)

    elif model_name == 'SwinIR':
        output_path = Path(f"{output_dir}{input_path.replace(".png", "_sr.pt")}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.open(full_path)
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Grayscale()])
        img = transform(img)[0]
        print(img.shape)
        output, _ = upsampler.enhance(np.array(img, dtype=np.float32), outscale = 4)
        print(output.shape)
        #for i in range(output.shape[0]):
        #    output[i, :, :] = output_slice

    torch.save(output, output_path)
    #cv2.imwrite(output_path, output)
    print(f"Saved {output_path}\n\n")

print("All images processed.")

from bisenet import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image, ImageOps
import cv2
import torchvision.transforms as transforms


# save segmentation result
def segmentation(im, mask, save_path):
    # Colors for beard and mustache
    part_colors = [[0, 255, 0], [0, 0, 255]]

    vis_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_of_class = np.max(mask)

    for pi in range(1, num_of_class + 1):
        index = np.where(mask == pi)
        vis_mask[index[0], index[1], :] = part_colors[pi - 1]

    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    vis_im = cv2.addWeighted(im, 1, vis_mask, 0.15, 0)
    cv2.imwrite(save_path, vis_im)


if __name__ == "__main__":
    save_path = 'seg_results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    net = BiSeNet(n_classes=3)
    model_pth = '../models/model-seg.pth'
    checkpoint = torch.load(model_pth)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    im_folder = '../examples'
    examples = os.listdir(im_folder)
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for i in examples:
            img = Image.open(osp.join(im_folder, i))
            img = ImageOps.exif_transpose(img)
            img = img.resize((512, 512), Image.BILINEAR)
            im = to_tensor(img)
            im = torch.unsqueeze(im, 0)
            out, out16, out32 = net(im)
            out = out.squeeze(0).cpu().numpy().argmax(0)
            out_path = i[:-3] + 'png'
            segmentation(img, out, osp.join(save_path, out_path))

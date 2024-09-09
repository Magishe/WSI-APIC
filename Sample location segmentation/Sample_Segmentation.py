import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import supervision as sv
import scipy.signal as ss

def Calculate_Boundary(img):
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r,b = cv2.threshold(g,100,255,cv2.THRESH_BINARY)


    inten_h = np.mean(b,1)
    mean_h = np.mean(inten_h)
    max_h = np.max(inten_h)
    peaks_h = ss.find_peaks(inten_h, height = (mean_h+max_h)/2,prominence = (mean_h+max_h)/3, width = np.array([0,50]))

    h1 = peaks_h[0][0]
    h2 = peaks_h[0][-1]


    inten_h1 = b[h1+50,:]
    l1 = ss.find_peaks(inten_h1)[0][0]
    r2, b2 = cv2.threshold(g[:,1600:2592], 120, 255, cv2.THRESH_BINARY)
    inten_l= np.mean(b2,0)
    mean_l = np.mean(inten_l)
    max_l = np.max(inten_l)
    peaks_l_2 = ss.find_peaks(inten_l, height = (mean_l+max_l)/4, prominence = (mean_l+max_l)/4,width = np.array([0,20]))
    l2 = peaks_l_2[0][-1]+1600
    return h1,h2,l1,l2



image_file = "NSCLC.tif"
img_ori = cv2.imread(image_file)


h1,h2,l1,l2 = Calculate_Boundary(img_ori)

## Please download this
checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam.to(device=DEVICE)


img = img_ori[h1:h2,l1:l2+9,:]
mask_generator = SamAutomaticMaskGenerator(sam)
result = mask_generator.generate(img)

filtered_results = [item for item in result if item['area'] >= 3000]
sorted_results = sorted(filtered_results, key=lambda x: x['area'],  reverse=True)
image_1 = np.multiply(img[:, :, 0], sorted_results[1]['segmentation'])
sample_mean_score1 = np.mean(np.mean(image_1[image_1>0]))
sample_std_score1 = np.std(image_1[image_1 > 0])

zero_array = np.zeros(np.shape(img))

annotated_mask = np.zeros(np.shape(img[:,:,0])).astype('bool')
cross_mask = np.zeros(np.shape(img[:,:,0])).astype('bool')

for i in range(1,len(sorted_results)):
    zeroone_binary = sorted_results[i]['segmentation']
    image_i = np.multiply(img[:,:,0], zeroone_binary)
    sample_mean_score = np.mean(np.mean(image_i[image_i>0]))
    sample_std_score = np.std(image_i[image_i > 0])
    iou = sorted_results[i]['predicted_iou']
    if (sample_mean_score > 0.75*sample_mean_score1) & (sample_std_score<sample_std_score1*1.05) & (iou>0.95):
        if sorted_results[i]['point_coords'][0][0] > np.shape(img[:,:,0])[1]*0.9:
            zeroone_binary = zeroone_binary.astype('bool')
            cross_mask = cross_mask | zeroone_binary
        else:
            zeroone_binary = zeroone_binary.astype('bool')
            annotated_mask = annotated_mask|zeroone_binary

annotated_mask = annotated_mask.astype('uint8')
cross_mask = cross_mask.astype('uint8')
mask_extend_value = 20
kernel = np.ones((mask_extend_value, mask_extend_value))
dilated_mask = cv2.dilate(annotated_mask, kernel, iterations=1)
binary = dilated_mask*255
binary_cross = cross_mask*255
plt.imshow(binary)
plt.show()




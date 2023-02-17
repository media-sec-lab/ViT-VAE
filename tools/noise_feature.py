import cv2
import numpy as np
from tools.SRM_noise import SRM
def feature_concat(img,noise):
    ## SRM_feature  img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    noise_srm1 = SRM([img_cv])
    noise_srm1 = np.transpose(noise_srm1, (1, 2, 0))
    noise_srm1 = cv2.cvtColor(noise_srm1.astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255
    noise_srm1 = noise_srm1[np.newaxis,:,:]


    ## laplas_feature
    lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    lap = cv2.cvtColor(lap.astype(np.uint8), cv2.COLOR_BGR2GRAY)/255
    lap = lap[np.newaxis, :, :]

    ## noiseprint
    noise = noise[np.newaxis, :, :]

    feature = np.concatenate([noise, noise_srm1, lap], axis=0)

    return feature
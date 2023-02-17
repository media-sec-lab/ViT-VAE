# ViT-VAE
This repo is the official implementation of "Learning Traces by Yourself: Blind Image Forgery
Localization via Anomaly Detection with ViT-VAE". <br>

<p align='center'>  
  <img src='https://github.com/chenyan764/ViT-VAE/blob/main/ViT-VAE.png' width='550'/>
</p>

# Installation
The code requires Python 3.7 and PyTorch 1.7.

# Usage
You can run test.py to generate the predicted results of the test image.<br>

For testing:

`
python test.py 
`

img_path: Path of the image. 

noise_path: Path of Noiseprint feature. <br>

Note: ViT-VAE needs to use the Noiseprint feature. <br>
You need to replace `main_extraction.py ` with `.\tools\main_extraction_ViT-VAE.py ` in [Noiseprint](https://github.com/grip-unina/noiseprint). <br>
This code is used to generate the noise map of the image.


save_path: Save path of prediction results (not thresholded).


mask_path: Ground-truth of the image.

# Acknowledgments
[Noiseprint: https://github.com/grip-unina/noiseprint](https://github.com/grip-unina/noiseprint)
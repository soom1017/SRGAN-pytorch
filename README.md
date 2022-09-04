# SRGAN-pytorch

SRGAN(CVPR 2017) pytorch implementation

Implementation of CVPR2017 Paper: ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802) in PyTorch

### Performance
- Trained my SRGAN model on DIV2K images
- Train process is divided into two steps, referred to [original implementation](https://github.com/tensorlayer/srgan)
  - `trainforinit`: initialize learning (G) w/ 10 epochs
  - `train`: adversarial learning (G, D) w/ 40 epochs
  
### Sample outputs
- Tested with Set5(x4 LR bicubic) images
- You can get sample output images in `examples/`

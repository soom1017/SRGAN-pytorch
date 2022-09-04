import torch.nn as nn
from torchvision.models import vgg19
import config

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # phi_5,4 5th conv layer before maxpooling but after activation
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, fake, target):
        vgg_fake_features = self.vgg((fake + 1) / 2.)
        vgg_target_features = self.vgg((target + 1) / 2.)
        return self.loss(vgg_fake_features, vgg_target_features)
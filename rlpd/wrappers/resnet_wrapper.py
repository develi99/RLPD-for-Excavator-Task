# PyTorch ResNet-18 forward (vereinfacht)
# x = self.conv1(x)
# x = self.bn1(x)
# x = self.relu(x)
# x = self.maxpool(x)

# x = self.layer1(x)
# x = self.layer2(x)
# x = self.layer3(x)
# x = self.layer4(x)

# x = self.avgpool(x)   # -> 1x1x512
# x = torch.flatten(x,1) # -> 512
# x = self.fc(x)        # -> 1000


import gym
import collections
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from gym.spaces import Box
import cv2

class VisionResNetWrapper(gym.ObservationWrapper):
    def __init__(self, env, device="cpu", resnet_layer="layer3", img_size=224):
        """
        resnet_layer: "layer2" -> 28x28, "layer3" -> 14x14, "layer4" -> 7x7
        """
        super().__init__(env)
        self.device = device
        self.resnet_layer = resnet_layer
        self.img_size = img_size

        # --- ResNet vorbereiten ---
        self.cnn = self._make_resnet_feature_extractor(resnet_layer).to(device)
        self.cnn.eval()

        # --- Preprocessing ---
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # --- Observation Space anpassen ---
        obs_space = env.observation_space
        low_dim_size = 3 + obs_space["stone"].shape[0] if "stone" in obs_space.spaces else 3
        # Featuremap Channels nach ResNet Layer
        channels = {"layer2": 128, "layer3": 256, "layer4": 512}[resnet_layer]
        feature_map_size = {"layer2": 28, "layer3": 14, "layer4": 7}[resnet_layer]

        # Flatten Featuremap
        image_feat_size = channels * feature_map_size * feature_map_size

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(low_dim_size + image_feat_size,),
            dtype=np.float32
        )

    def _make_resnet_feature_extractor(self, stop_at_layer="layer3"):
        """Minimaler ResNet-18, stoppt nach gewünschter Layer, kein FC/GAP"""
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.stop_at_layer = stop_at_layer

        class FeatureExtractor(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.stem = outer.stem
                self.layer1 = outer.layer1
                self.layer2 = outer.layer2
                self.layer3 = outer.layer3
                self.layer4 = outer.layer4
                self.stop_at_layer = outer.stop_at_layer

            def forward(self, x):
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                if self.stop_at_layer == "layer2":
                    return x
                x = self.layer3(x)
                if self.stop_at_layer == "layer3":
                    return x
                x = self.layer4(x)
                return x

        return FeatureExtractor(self)

    def observation(self, obs):
        # --- Low-Dim Features ---
        low_dim = np.concatenate([
            obs["policy"].flatten()[:3],
            obs["stone"].flatten() if "stone" in obs else np.array([], dtype=np.float32)
        ])

        # --- Image ---
        img = obs["camera"]["rgb"]
        if img.shape[0] != 3:
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        img = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_map = self.cnn(img)  # BxCxHxW
            img_feat = feat_map.flatten(1).cpu().numpy().squeeze(0)

        # --- Kombiniere ---
        return np.concatenate([low_dim, img_feat]).astype(np.float32)
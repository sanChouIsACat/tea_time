import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from util.serializer import serialiable


@dataclass
@serialiable
class VaeCtx:
    input_dim: int
    encoder_hidden_dim: list[int]
    latent_dim: int
    decoder_hidden_dim: list[int]


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, ctx: VaeCtx):
        super(VAE, self).__init__()
        self.ctx = ctx

        # 编码器
        self.encoder = self.build_mlp(
            self.ctx.input_dim, self.ctx.encoder_hidden_dim, self.ctx.latent_dim
        )

        # 解码器
        self.decoder = self.build_mlp(
            self.ctx.latent_dim, self.ctx.decoder_hidden_dim, self.ctx.input_dim
        )

        # 均值和对数方差
        self.fc_mu = nn.Linear(self.ctx.encoder_hidden_dim[-1], self.ctx.latent_dim)
        self.fc_logvar = nn.Linear(self.ctx.encoder_hidden_dim[-1], self.ctx.latent_dim)

    def build_mlp(self, input_dim, hidden_dims, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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
    output_dim: int
    residual_depth: int = 3


class BasicMlp(nn.Module):
    def __init__(
        self,
        inputDim: int,
        outputDim: int,
        non_linear_f: nn.Module,
        normalization_f: nn.Module,
    ):
        super().__init__()
        self.__linear_f = nn.Linear(inputDim, outputDim)
        self.__non_linear_f = non_linear_f
        self.__normalization_f = normalization_f

    def forward(self, x: torch.Tensor):
        x = self.__linear_f(x)
        x = self.__non_linear_f(x)
        x = self.__normalization_f(x)
        return x


class Coder(nn.Module):
    def __init__(
        self,
        hidden_dims: list[int],
        input_dim: int,
        output_dim: int,
        residual_depth: int,
    ):
        super().__init__()
        self.__residual_depth = residual_depth
        self.__transform_to_hidden_f = BasicMlp(
            input_dim, hidden_dims[0], nn.Sigmoid(), nn.LayerNorm(hidden_dims[0])
        )

        hidden_layers = []
        for i in range(1, len(hidden_dims)):
            hidden_layers.append(
                BasicMlp(
                    hidden_dims[i - 1],
                    hidden_dims[i],
                    nn.Sigmoid(),
                    nn.LayerNorm(hidden_dims[i]),
                )
            )
        self.__hidden_fs = nn.ModuleList(hidden_layers)

        self.__transform_to_output_f = BasicMlp(
            hidden_dims[0], output_dim, nn.Sigmoid(), nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor):
        x = self.__transform_to_hidden_f(x)
        origin = x.clone()
        current_depth = 0
        for layer in self.__hidden_fs:
            current_depth = (current_depth + 1) % self.__residual_depth
            x = layer(x)
            if current_depth == 0:
                x = x + origin
                origin = x.clone()
        return self.__transform_to_output_f(x)


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, ctx: VaeCtx):
        super(VAE, self).__init__()
        self.ctx = ctx

        # encoder/decoder
        self.encoder = Coder(
            self.ctx.encoder_hidden_dim[:-1],
            self.ctx.input_dim,
            self.ctx.encoder_hidden_dim[-1],
            self.ctx.residual_depth,
        )
        self.decoder = Coder(
            self.ctx.decoder_hidden_dim,
            self.ctx.latent_dim,
            self.ctx.output_dim,
            self.ctx.residual_depth,
        )

        # 均值和对数方差
        self.fc_mu = nn.Linear(self.ctx.encoder_hidden_dim[-1], self.ctx.latent_dim)
        self.fc_logvar = nn.Linear(self.ctx.encoder_hidden_dim[-1], self.ctx.latent_dim)

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = self.reparameterize(mu, torch.zeros_like(mu))
        return self.decode(z), mu, logvar

    def loss_f(
        self,
        output: torch.Tensor,
        label: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recons_loss = F.mse_loss(output, label)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1),
            dim=0,
        )

        loss = recons_loss + kld_loss
        return loss, recons_loss, kld_loss

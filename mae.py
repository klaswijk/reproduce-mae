import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.vision_transformer import EncoderBlock


class Transformer(nn.Module):
    """
    Transformer Model for sequence to sequence translation.
    Based on torchvision.models.vision_transformer.Encoder
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        return self.ln(self.layers(self.dropout(input)))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, seq_len: int, token):
        super().__init__()
        height = width = int(math.sqrt(seq_len - 1 if token else seq_len))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(
            pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(
            pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(
            pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        pe = pe.reshape(d_model * 2, seq_len - 1 if token else seq_len)
        if token:
            pe = torch.cat([torch.zeros((pe.shape[0], 1)), pe], dim=1)
        pe = pe.permute(1, 0)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x


class MAE(nn.Module):
    """
    Masked autoencoder
    """

    def __init__(
        self,
        image_size: int,
        n_classes: int,
        patch_size: int,
        encoder_layers: int,
        encoder_hidden_dim: int,
        encoder_num_heads: int,
        encoder_mlp_dim: int,
        decoder_layers: int,
        decoder_hidden_dim: int,
        decoder_num_heads: int,
        decoder_mlp_dim: int,
        mask_ratio: float,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.seq_length = (image_size // patch_size) ** 2
        self.mask_length = int((1 - mask_ratio) * self.seq_length)

        # Encoder
        self.encoder_norm = nn.LayerNorm(encoder_hidden_dim)
        self.encoder_proj = nn.Conv2d(
            3, encoder_hidden_dim, patch_size, patch_size)
        self.encoder_pos_encoding = PositionalEncoding(
            encoder_hidden_dim, self.seq_length + 1, True)
        self.encoder = Transformer(
            encoder_layers, encoder_num_heads, encoder_hidden_dim, encoder_mlp_dim)

        # Decoder
        self.decoder_norm = nn.LayerNorm(decoder_hidden_dim)
        self.hidden_proj = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.decoder_pos_encoding = PositionalEncoding(
            decoder_hidden_dim, self.seq_length, False)
        self.decoder = Transformer(
            decoder_layers, decoder_num_heads, decoder_hidden_dim, decoder_mlp_dim)
        self.decoder_proj = nn.Linear(decoder_hidden_dim, 3 * patch_size**2)

        # Classifier
        self.class_token = nn.Parameter(torch.zeros(1, 1, encoder_hidden_dim))
        self.classifier = nn.Linear(encoder_hidden_dim, n_classes)

        # Init
        w = self.encoder_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def patch(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_proj(x)
        x = x.reshape(x.shape[0], self.encoder_hidden_dim,
                      (self.image_size // self.patch_size)**2)
        x = x.permute(0, 2, 1)
        return x

    def unpatch(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_proj(x)
        x = x.permute(0, 2, 1)
        x = F.fold(x, (self.image_size, self.image_size), (self.patch_size,
                   self.patch_size), stride=self.patch_size)
        return x

    def mask(self, x: torch.Tensor) -> tuple:
        perm = torch.randperm(self.seq_length) + 1
        # Class token at index 0
        perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])
        x = x[:, perm]
        masked = x[:, :self.mask_length + 1]
        return masked, perm

    def unmask(self, x: torch.Tensor, masked: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        unshuf = torch.zeros_like(perm)
        unshuf[perm] = torch.arange(self.seq_length + 1)
        out = x[:, unshuf]
        out[:, perm[1:self.mask_length + 1]] = 0
        return x

    def encoder_forward(self, x: torch.Tensor, mask: bool) -> tuple:
        x = self.patch(x)
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.encoder_pos_encoding(x)
        if mask:
            masked, perm = self.mask(x)
            masked = self.encoder(masked)
            masked = self.encoder_norm(masked)
            x = self.unmask(x, masked, perm)
            # Don't include the class token in perm
            return x, perm[1:self.mask_length + 1] - 1
        else:
            x = self.encoder(x)
            x = self.encoder_norm(x)
            return x, None

    def decoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_pos_encoding(x)
        x = self.decoder(x)
        x = self.decoder_norm(x)
        x = self.unpatch(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, masked_indices = self.encoder_forward(x, True)
        x = x[:, 1:]  # Remove class token
        x = self.hidden_proj(x)
        x = self.decoder_forward(x)
        return x, masked_indices

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.encoder_forward(x, False)
        x = x[:, 0]  # Extract class token
        x = self.classifier(x)
        return x

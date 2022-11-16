import os
import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from torchvision.models.vision_transformer import EncoderBlock


class Transformer(nn.Module):
    """Based on torchvision.models.vision_transformer.Encoder
       Transformer Model for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
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
    """Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:d
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class MAE(nn.Module):

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
        self.seq_length = (image_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.mask_length = int((1 - self.mask_ratio) * self.seq_length)
        self.encoder_pos_encoding = PositionalEncoding(encoder_hidden_dim, self.seq_length + 1)
        self.decoder_pos_encoding = PositionalEncoding(decoder_hidden_dim, self.seq_length)

        self.encoder = Transformer(
            self.mask_length + 1,
            encoder_layers,
            encoder_num_heads,
            encoder_hidden_dim,
            encoder_mlp_dim,
            0,
            0
        )
        self.decoder = Transformer(
            self.seq_length,
            decoder_layers,
            decoder_num_heads,
            decoder_hidden_dim,
            decoder_mlp_dim,
            0,
            0
        )
        self.patch_size = patch_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.encoder_hidden_dim))

        self.encoder_conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=encoder_hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.hidden_proj = nn.Linear(
            in_features=encoder_hidden_dim,
            out_features=decoder_hidden_dim,
        )
        self.decoder_inv_proj = nn.Linear(
            in_features=decoder_hidden_dim,
            out_features=3*patch_size**2,
        )
        self.classifier = nn.Linear(
            in_features=encoder_hidden_dim,
            out_features=n_classes
        )

    def patches(self, x):
        n, _, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size,
                      f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size,
                      f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # Linear embedding
        x = self.encoder_conv_proj(x)
        x = x.reshape(n, self.encoder_hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return x

    def encoder_forward(self, x):
        x = self.patches(x)

        # Prepend class token
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Add positional embedding
        x = self.encoder_pos_encoding(x)

        # Shuffle
        perm = torch.randperm(self.seq_length) + 1
        perm = torch.cat([torch.zeros(1, dtype=torch.long), perm])  # Always place the class token first
        x = x[:, perm]
        masked = x[:, :self.mask_length + 1]

        # Encode
        masked = self.encoder(masked)

        # Unshuffle
        x = torch.zeros_like(x)
        x[:, perm[:self.mask_length + 1]] = masked
        return x, perm[1:self.mask_length + 1] - 1 # Don't include the class token in perm

    def decoder_forward(self, x):
        # Decode
        x = self.decoder_pos_encoding(x)
        x = self.decoder(x)

        # Project back to image
        n = x.shape[0]
        h = w = self.image_size
        x = self.decoder_inv_proj(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(n, 3, h, w)
        return x

    def forward(self, x):
        x, masked_indices = self.encoder_forward(x)
        x = x[:, 1:]  # Remove class token
        x = self.hidden_proj(x)  # Linear projection 
        x = self.decoder_forward(x)
        return x, masked_indices

    def unmasked_encoder_forward(self, x):
        x = self.patches(x)

        # Prepend class token
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Add positional embedding
        x = self.encoder_pos_encoding(x)

        # Encode
        x = self.encoder(x)

        return x[:, 0]

    def classify(self, x):
        x = self.unmasked_encoder_forward(x)
        return self.classifier(x)


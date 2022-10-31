import torch
from torch import nn
from torchvision.models.vision_transformer import Encoder


class Masker(nn.Module):
    
    def __init__(
        self,
        mask_ratio: float,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio


    def forward(self, x):
        return NotImplemented


class MAE(nn.Module):

    def __init__(
        self,
        seq_length: int,
        encoder_layers: int,
        encoder_,
        decoder_layers: int,
        mask_ratio: float,

    ):
        super().__init__()
        self.masker = Masker()
        self.encoder = Encoder()
        self.decoder = Encoder()


    def patches(self, x):
        n, _, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return x


    def encoder_forward(self, x):
        # Apply tokens

        # Random mask

        # Concat

        # Shuffle

        # Encode

        # Deshuffle

        return NotImplemented


    def decoder_forward(self, x):
        # Sparse

        # Decode

        return NotImplemented


    def forward(self, x):
        return self.decoder_forward(self.encoder_forward(x))

    
    def classify(self, x):
        return self.classifier(self.encoder_forward(x))


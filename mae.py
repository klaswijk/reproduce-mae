import torch
from torch import nn
from torchvision.models.vision_transformer import Encoder as Transformer


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
        image_size: int,
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

        self.seq_length = (image_size // patch_size) ** 2

        self.mask_length = int((1 - mask_ratio) * self.seq_length)

        self.masker = Masker()
        self.encoder = Transformer(
            self.seq_length - self.mask_length,
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

    def patches(self, x):
        n, _, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # Linear embedding?
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        x = x.permute(0, 2, 1)
        return x

    def encoder_forward(self, x):
        # Apply tokens

        x = self.patches(x)

        perm = torch.randperm(self.seq_length)

        x = x[perm]
        masked = x[:self.mask_length]

        masked = self.encoder(masked)

        # Deshuffle
        x = torch.zeros(self.seq_length)

        x[perm[:self.mask_length]] = masked

        return x

    def decoder_forward(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder_forward(self.encoder_forward(x))

    def classify(self, x):
        return self.classifier(self.encoder_forward(x))

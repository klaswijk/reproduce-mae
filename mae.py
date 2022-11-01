import torch
from torch import nn
from torchvision.models.vision_transformer import Encoder as Transformer


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
        self.image_size = image_size
        self.seq_length = (image_size // patch_size) ** 2
        self.mask_length = int((1 - mask_ratio) * self.seq_length)
        self.encoder = Transformer(
            self.mask_length,
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
        self.encoder_conv_proj = nn.Conv2d(
            in_channels=3, out_channels=encoder_hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.decoder_hidden_dim = encoder_hidden_dim
        self.decoder_inv_conv_proj = nn.ConvTranspose2d(
            in_channels=decoder_hidden_dim, out_channels=3, kernel_size=patch_size, stride=patch_size
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
        perm = torch.randperm(self.seq_length)
        x = x[:, perm]
        masked = x[:, :self.mask_length]

        masked = self.encoder(masked)

        x = torch.zeros_like(x)
        x[:, perm[:self.mask_length]] = masked
        return x

    def decoder_forward(self, x):
        n = x.shape[0]
        h = w = self.image_size // self.patch_size

        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(n, self.decoder_hidden_dim, h, w)
        x = self.decoder_inv_conv_proj(x)
        return x

    def forward(self, x):
        return self.decoder_forward(self.encoder_forward(x))

    def classify(self, x):
        return self.classifier(self.encoder_forward(x))

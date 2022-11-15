import os
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

    def classify(self, x):
        return self.classifier(self.encoder_forward(x))


def small_model(image_size, patch_size, mask_ratio, weight_path=None):
    model = MAE(
        image_size=image_size,
        patch_size=patch_size,
        encoder_layers=8,
        encoder_num_heads=8,
        encoder_hidden_dim=256,
        encoder_mlp_dim=1024,
        decoder_layers=4,
        decoder_num_heads=8,
        decoder_hidden_dim=64,
        decoder_mlp_dim=256,
        mask_ratio=mask_ratio,
    )
    if weight_path:
        print(f"Loading model from '{weight_path}'... ", end='')
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print("Load successful!")
        else:
            print(f"No weights found")
    return model
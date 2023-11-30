'''

'''

import torch
import torch.nn as nn

class ConditionalAutoencoder3D(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(ConditionalAutoencoder3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x, condition):
        encoded = self.encoder(x)
        encoded_condition = self.encoder(condition)
        encoded_concat = torch.cat([encoded, encoded_condition], dim=1)
        decoded = self.decoder(encoded_concat)
        return decoded

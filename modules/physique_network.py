import torch.nn as nn
import torch

class PhysiqueMaskGenerator(nn.Module):
    def __init__(self, num_features, num_parts=1):
        super(PhysiqueMaskGenerator, self).__init__()
        """
        convolutional enncoder, decoder
        """
        self.num_features = num_features
        self.num_parts = num_parts

        self.encoder, self.decoder = self.define_network(self.num_features)

    def _add_conv_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.SyncBatchNorm(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_down_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 2, 1),
            nn.SyncBatchNorm(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_up_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.SyncBatchNorm(nf),
            nn.LeakyReLU(inplace=True)
        )

    def define_network(self, num_features):
        # encoder
        encoder = [self._add_conv_layer(in_ch=self.num_parts, nf=num_features[0])]
        for i in range(1, len(num_features)):
            encoder.append(self._add_conv_layer(num_features[i - 1], num_features[i - 1]))
            encoder.append(self._add_down_layer(num_features[i - 1], num_features[i]))

        # decoder mirrors the encoder
        decoder = []
        for i in range(len(num_features)-1, 0, -1):
            decoder.append(self._add_conv_layer(num_features[i], num_features[i]))
            decoder.append(self._add_up_layer(num_features[i], num_features[i-1]))

        decoder.append(nn.Conv2d(num_features[i-1], 1, 3, 1, 1))
        return nn.Sequential(*encoder), nn.Sequential(*decoder)

    def forward(self, input):

        x = self.encoder(input)
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x

import torch
import torch.nn as nn



class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        if cfg.MODEL.BACKBONE.BASIC:
            self.basic_model_layers(image_channels)
            # self.blocks = [
            #     *self.basic_model_layers(image_channels, self.output_channels)
            # ]
        else:
            self.best_model_layers(image_channels)
            # self.blocks = [
            #     *self.best_model_layers(image_channels, self.output_channels)
            # ]

    def best_model_layers(self, image_channels):
        self.first_block = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.output_channels[0], 3, stride=2, padding=1)
        )
        self.second_block = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[0]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[0], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[1], 3, stride=2, padding=1)
        )
        self.third_block = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[2], 3, stride=2, padding=1)
        )
        self.fourth_block = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[2], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[3], 3, stride=2, padding=1)
        )
        self.fifth_block = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[3], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[4], 3, stride=2, padding=1)
        )
        self.sixth_block = nn.Sequential(
            nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
            nn.Conv2d(self.output_channels[4], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[5], 3, padding=0)
        )
        self.blocks = [self.first_block, self.second_block, self.third_block, self.fourth_block, self.fifth_block, self.sixth_block]


    def basic_model_layers(self, image_channels):
        self.first_block = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.output_channels[0], 3, stride=2, padding=1)
        )
        self.second_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[0], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[1], 3, stride=2, padding=1)
        )
        self.third_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[1], 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[2], 3, stride=2, padding=1)
        )
        self.fourth_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[2], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[3], 3, stride=2, padding=1)
        )
        self.fifth_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[3], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[4], 3, stride=2, padding=1)
        )
        self.sixth_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[4], 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[5], 3, padding=0)
        )
        self.blocks = [self.first_block, self.second_block, self.third_block, self.fourth_block, self.fifth_block, self.sixth_block]

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for block in self.blocks:
            out_features.append(block(x))
            x = out_features[-1]
        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            # NOTE: I changed out_channel to self.output_channels[idx] as I do believe this is correct.
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

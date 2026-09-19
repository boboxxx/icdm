import torch.nn as nn

# from loss.distortion import Distortion
# from Autoencoder.loss.distortion import Distortion

from Autoencoder.SwinJSCC.decoder import *
from Autoencoder.SwinJSCC.encoder import *
from Autoencoder.MambaJSCC.decoder import *
from Autoencoder.MambaJSCC.encoder import *
from Autoencoder.DeepJSCC.decoder import *
from Autoencoder.DeepJSCC.encoder import *


class JSCC_encoder(nn.Module):
    def __init__(self, config):
        super(JSCC_encoder, self).__init__()
        self.create_model_dict = {
            "SwinJSCC": create_SwinJSCC_encoder,
            "MambaJSCC": create_MambaJSCC_encoder,
            "DeepJSCC": create_DeepJSCC_encoder,
        }

        self.encoder = self.create_model_dict.get(config.MODEL.MODEL_NAME, None)(config)

    def forward(self, input_image):
        feature = self.encoder(input_image)

        return feature


class JSCC_decoder(nn.Module):
    def __init__(self, config):
        super(JSCC_decoder, self).__init__()
        self.create_model_dict = {
            "SwinJSCC": create_SwinJSCC_decoder,
            "MambaJSCC": create_MambaJSCC_decoder,
            "DeepJSCC": create_DeepJSCC_decoder,
        }

        self.decoder = self.create_model_dict[config.MODEL.MODEL_NAME](config)

    def forward(self, feature):
        recon_image = self.decoder(feature)

        return recon_image


class JSCC_infencoder(JSCC_encoder):
    def __init__(self, config):
        super(JSCC_infencoder, self).__init__(config)

        self.encoder = self.create_model_dict.get(config.MODEL.INF_MODEL_NAME, None)(config)

    def forward(self, input_image):
        feature = self.encoder(input_image)

        return feature


class JSCC_infdecoder(JSCC_decoder):
    def __init__(self, config):
        super(JSCC_infdecoder, self).__init__(config)

        self.decoder = self.create_model_dict.get(config.MODEL.INF_MODEL_NAME, None)(config)

    def forward(self, input_image):
        feature = self.decoder(input_image)

        return feature

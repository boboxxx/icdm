# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
_C.MODEL = CN()
_C.TRAIN = CN()
_C.DATA = CN()
_C.CHANNEL = CN()
##################### default config #####################
######## model config ########
_C.MODEL.PATCH_SIZE = 1
_C.MODEL.IN_CHANS = 1
_C.MODEL.EMBED_DIMS = [1]
_C.MODEL.DEPTHS = [1]
_C.MODEL.NUM_HEADS = [1]
_C.MODEL.WINDOW_SIZE = 8
_C.MODEL.MLP_RATIO = 4.0
_C.MODEL.OUT_CHANS = 4

_C.MODEL.MODEL_NAME = ""
_C.MODEL.INF_MODEL_NAME = ""
######## save path ########
_C.MODEL.ENCODER_PATH = ""
_C.MODEL.DECODER_PATH = ""
_C.MODEL.INFCODE_PATH = ""
_C.MODEL.INFDECODER_PATH = ""
_C.MODEL.ICDM_S_PATH = ""
_C.MODEL.ICDM_Z_PATH = ""

_C.MODEL.DNCNN_PATH = ""
_C.MODEL.DIC_PATH = ""

_C.MODEL.LAMBDA = 1.5
_C.MODEL.BETA = 1.0
######## train environment config ########
_C.TRAIN.EPOCHS = 1
_C.TRAIN.SAVE_FRE = 1
_C.TRAIN.ICDM_EPOCHS = 1
_C.TRAIN.ICDM_SAVE_FRE = 1
_C.TRAIN.ICDM_TEST_FRE = 1
_C.TRAIN.LOSS = "LPIPS"
_C.TRAIN.EVAL_LOSS = "LPIPS"
_C.TRAIN.TRAIN_PRINT_FRE = 10  # PRINT FREQUENCY EACH EPOCH
_C.TRAIN.DIC_EPOCHS = 1
_C.TRAIN.DnCNN_EPOCHS = 1
######## data config ########
_C.DATA.DATASET = "DIV2K"
_C.DATA.INFDATA = "CIFAR10"
_C.DATA.TRAIN_BATCH = 1
_C.DATA.TRAIN_ICDM_BATCH = 1
_C.DATA.TEST_BATCH = 1
_C.DATA.IMG_SIZE = 256
_C.DATA.TRAIN_PATH = ""
_C.DATA.TEST_PATH = ""
_C.DATA.INF_TRAIN_PATH = ""
_C.DATA.INF_TEST_PATH = ""

## channel config ##
_C.CHANNEL.TRAIN_SNR = 1
_C.CHANNEL.EVAL_SNR = 1
_C.CHANNEL.TRAIN_SINR = 1
_C.CHANNEL.EVAL_SINR = 1
_C.CHANNEL.TYPE = "awgn"

#################### fix config ####################
_C.LEARNING_RATE = 1e-4


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
            print(1)
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    # config.freeze()


def update_config(config, args):

    _update_config_from_file(config, args.model_config_path)
    _update_config_from_file(config, args.train_config_path)


def get_config(args):
    config = _C.clone()
    model_config_path = f"./configs/models/{args.model}_{args.depths}.yaml"
    train_config_path = f"./configs/train/{args.dataset}.yaml"
    _update_config_from_file(config, model_config_path)
    _update_config_from_file(config, train_config_path)

    if args.mode == "train_inf":
        train_config_path = f"./configs/train/{args.interference_dataset}.yaml"

    _update_config_from_file(config, train_config_path)

    config.DATA.DATASET = args.dataset
    config.DATA.INFDATA = args.interference_dataset

    config.TRAIN.LOSS = args.loss
    config.TRAIN.EVAL_LOSS = args.eval_loss

    config.MODEL.OUT_CHANS = args.C
    config.MODEL.INF_MODEL_NAME = args.inf_model
    config.CHANNEL.TYPE = args.channel_type
    config.CHANNEL.TRAIN_SNR = args.train_SNR
    config.CHANNEL.EVAL_SNR = args.eval_SNR
    config.CHANNEL.TRAIN_SINR = args.train_SINR
    config.CHANNEL.EVAL_SINR = args.eval_SINR
    config.DATA.INF_IMG_SIZE = {"CIFAR10": 32, "CelebA": 128}[args.interference_dataset]
    config.TrainFromCheckpoint = args.train_from_checkpoint
    basepath = f"/mnt/sda/wt/JDMcheckpoints"
    config.MODEL.ENCODER_PATH = (
        basepath
        + f"/JSCC/{args.dataset}/{args.model}/{args.loss}/ENCODER_SNR{args.train_SNR}_SINR{args.train_SINR}_OUTCHANS{args.C}_DEPTHS{args.depths}_CHANNEL{args.channel_type}.pth"
    )
    config.MODEL.DECODER_PATH = (
        basepath
        + f"/JSCC/{args.dataset}/{args.model}/{args.loss}/DECODER_SNR{args.train_SNR}_SINR{args.train_SINR}_OUTCHANS{args.C}_DEPTHS{args.depths}_CHANNEL{args.channel_type}.pth"
    )
    config.MODEL.INFCODE_PATH = (
        basepath
        + f"/JSCC/{args.dataset}/{args.inf_model}/INFCODE{args.interference_dataset}_OUTCHANS{args.C}_DEPTHS{args.depths}.pth"
    )
    config.MODEL.INFDECODE_PATH = (
        basepath
        + f"/JSCC/{args.dataset}/{args.inf_model}/INFDECODE{args.interference_dataset}_OUTCHANS{args.C}_DEPTHS{args.depths}.pth"
    )
    config.MODEL.ICDM_S_PATH = (
        basepath
        + f"/ICDM/{args.dataset}/{args.model}/{args.loss}/ICDM_S_SNR{args.train_SNR}_SINR{args.train_SINR}_OUTCHANS{args.C}_DEPTHS{args.depths}_CHANNEL{args.channel_type}.pth"
    )
    config.MODEL.ICDM_Z_PATH = (
        basepath
        + f"/ICDM/{args.dataset}/{args.inf_model}/ICDM_Z{args.interference_dataset}_OUTCHANS{args.C}_DEPTHS{args.depths}_CHANNEL{args.channel_type}.pth"  # {args.channel_type}
    )
    config.MODEL.DNCNN_PATH = (
        basepath
        + f"/DnCNN/{args.dataset}/{args.inf_model}/DnCNN{args.interference_dataset}_OUTCHANS{args.C}_DEPTHS{args.depths}_CHANNEL{args.channel_type}.pth"  # {args.channel_type}
    )
    config.MODEL.DIC_PATH = (
        basepath
        + f"/DIC/{args.dataset}/{args.inf_model}/DIC{args.interference_dataset}_OUTCHANS{args.C}_DEPTHS{args.depths}_CHANNEL{args.channel_type}.pth"  # {args.channel_type}
    )
    config.LEARNING_RATE = args.lr
    if os.path.exists(basepath + f"/JSCC/{args.dataset}/{args.model}/{args.loss}") == False:
        os.makedirs(basepath + f"/JSCC/{args.dataset}/{args.model}/{args.loss}")
    if os.path.exists(basepath + f"/ICDM/{args.dataset}/{args.model}/{args.loss}") == False:
        os.makedirs(basepath + f"/ICDM/{args.dataset}/{args.model}/{args.loss}")

    if os.path.exists(basepath + f"/ICDM/{args.dataset}/{args.inf_model}") == False:
        os.makedirs(basepath + f"/ICDM/{args.dataset}/{args.inf_model}")

    if os.path.exists(basepath + f"/DnCNN/{args.dataset}/{args.inf_model}") == False:
        os.makedirs(basepath + f"/DnCNN/{args.dataset}/{args.inf_model}")
    if os.path.exists(basepath + f"/DIC/{args.dataset}/{args.inf_model}") == False:
        os.makedirs(basepath + f"/DIC/{args.dataset}/{args.inf_model}")
    return config


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary"""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

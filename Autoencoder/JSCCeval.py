import torch.optim as optim
import math
from Autoencoder.data.datasets import get_loader
from Autoencoder.channel import Channel
from Autoencoder.network import *
from utils import *
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

from datetime import datetime
import torch.nn as nn
import argparse
from Autoencoder.loss.distortion import *
import time
import sys
from tqdm import tqdm
import copy
from Autoencoder.loss.eval_matrix import loss_matrix, eval_matrix


# def load_weights(net, model_path):
#     pretrained = torch.load(model_path)  # .state_dict()  # ['state_dict']
#     result_dict = {}
#     for key, weight in pretrained.items():
#         result_key = key
#         if "attn_mask" not in key and "rate_adaption.mask" not in key:
#             result_dict[result_key] = weight
#     print(net.load_state_dict(result_dict, strict=False))
#     del result_dict, pretrained


@torch.no_grad()
def eval_only_JSCC(config, logger):
    seed_torch()
    logger.info(f"start eval JSCC")
    encoder = JSCC_encoder(config).cuda()
    decoder = JSCC_decoder(config).cuda()
    # if config.CHANNEL.TRAIN_SNR != config.CHANNEL.TRAIN_SINR:
    infcode = JSCC_infencoder(config).cuda()
    infcode_path = config.MODEL.INFCODE_PATH
    infcode.load_state_dict(torch.load(infcode_path))
    infcode.eval()
    print("inf encoder loaded")

    channel = Channel(config)

    encoder_path = config.MODEL.ENCODER_PATH
    decoder_path = config.MODEL.DECODER_PATH

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.eval()
    decoder.eval()

    _, test_loader, _, test_infloader = get_loader(config, need_padding=False)

    # test_mem_and_comp(config,encoder,decoder)
    performance = eval_matrix(config.TRAIN.EVAL_LOSS)

    eval_matrics, cmptime = [AverageMeter() for _ in range(2)]
    matrix = [eval_matrics, cmptime]
    SNR_list = Str_to_FloatList(config.CHANNEL.EVAL_SNR)
    SINR_list = Str_to_FloatList(config.CHANNEL.EVAL_SINR)
    for SNR in SNR_list:
        for SINR in SINR_list:
            assert SNR >= SINR, "min of SNR list should not smaller than max of SINR list"

            for element in matrix:
                element.clear()

            for i, ((input, label1), (inf, label2)) in enumerate(
                tqdm(zip(test_loader, test_infloader))
            ):

                start_time = time.time()
                input = input.cuda()
                inf = inf.cuda()
                upsample_ratio = input.shape[2] / inf.shape[2]
                inf = torch.nn.functional.interpolate(
                    inf, scale_factor=upsample_ratio, mode="nearest"
                )

                y = encoder(input)

                try:
                    inf_y = infcode(inf)
                except:
                    pass

                CBR = y.numel() / 2 / input.numel()

                if SNR == SINR:
                    noisy_y, pwr, h = channel.forward(y, SNR)
                else:
                    noisy_y, pwr, pwr_inf, h = channel.inf_forward(y, inf_y, SNR, SINR)

                if config.CHANNEL.TYPE in ["rayleigh"]:
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    # SNR 均衡，1. 匹配训练， 2 SINR是未知的，意外干扰带来的
                    noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
                elif config.CHANNEL.TYPE in ["awgn"]:
                    pass
                else:
                    raise ValueError

                noisy_y = noisy_y / math.sqrt(torch.mean(torch.abs(noisy_y) ** 2))
                noisy_y = torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2)

                recon_image = decoder(noisy_y)

                matric = performance(recon_image, input)

                cmptime.update(time.time() - start_time)
                eval_matrics.update(matric)

            log = " | ".join(
                [
                    f"SNR {SNR} SINR {SINR}",
                    f"Time_all {cmptime.sum:.2f}",
                    f"CBR {CBR:.4f}",
                    f"{config.TRAIN.EVAL_LOSS} {eval_matrics.val:.4f} ({eval_matrics.avg:.4f})",
                ]
            )
            logger.info(log)
            eval_matrics.append(eval_matrics.avg)

        if len(SINR_list) > 1:
            logger.info(f"SNR {SNR} SINR {SINR_list} ALL_Loss {eval_matrics.list}")
            eval_matrics.clear_list()
    if len(SNR_list) <= 1:
        logger.info(f"SNR {SNR_list} SINR {SINR} ALL_Loss {eval_matrics.list}")
        eval_matrics.clear_list()


@torch.no_grad()
def eval_inf_JSCC(config, logger):
    seed_torch()
    logger.info(f"start eval inf JSCC")

    infencoder = JSCC_infencoder(config).cuda()
    infdecoder = JSCC_infdecoder(config).cuda()
    infencoder_path = config.MODEL.INFCODE_PATH
    infdecoder_path = config.MODEL.INFDECODE_PATH
    # print(torch.load(infencoder_path))

    infencoder.load_state_dict(torch.load(infencoder_path))
    infdecoder.load_state_dict(torch.load(infdecoder_path))
    infencoder.eval()
    infdecoder.eval()

    channel = Channel(config)

    _, test_Loader, _, test_infloader = get_loader(config, need_padding=False)
    upsampling_ratio = next(iter(test_Loader))[0].shape[2] / next(iter(test_infloader))[0].shape[2]
    performance = eval_matrix(config.TRAIN.EVAL_LOSS)

    loader = test_infloader

    eval_matrics, cmptime = [AverageMeter() for _ in range(2)]
    matrix = [eval_matrics, cmptime]
    SNR_list = Str_to_FloatList(config.CHANNEL.TRAIN_SNR)

    for SNR in SNR_list:

        for element in matrix:
            element.clear()

        for i, (inf, label2) in enumerate(tqdm(loader)):

            start_time = time.time()

            inf = inf.cuda()

            inf = torch.nn.functional.interpolate(
                inf, scale_factor=upsampling_ratio, mode="nearest"
            )
            # print(inf.shape)
            y = infencoder(inf)

            CBR = y.numel() / 2 / inf.numel()

            noisy_y, pwr, h = channel.forward(y, SNR)

            if config.CHANNEL.TYPE in ["rayleigh"]:
                sigma_square = 1.0 / (10 ** (SNR / 10))
                noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
            elif config.CHANNEL.TYPE in ["awgn"]:
                pass
            else:
                raise ValueError

            noisy_y = noisy_y / math.sqrt(torch.mean(torch.abs(noisy_y) ** 2))
            noisy_y = torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2)

            recon_image = infdecoder(noisy_y)

            matric = performance(recon_image, inf)

            cmptime.update(time.time() - start_time)
            eval_matrics.update(matric)

        log = " | ".join(
            [
                f"SNR {SNR}",
                f"Time_all {cmptime.sum:.2f}",
                f"CBR {CBR:.4f}",
                f"{config.TRAIN.EVAL_LOSS} {eval_matrics.val:.4f} ({eval_matrics.avg:.4f})",
            ]
        )
        logger.info(log)
        eval_matrics.append(eval_matrics.avg)

    logger.info(f"SNR {SNR_list} ALL_Loss {eval_matrics.list}")
    eval_matrics.clear_list()

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


def train_JSCC_seqeratly(config, logger):
    seed_torch()
    logger.info(f"start train JSCC seqeratly")
    encoder = JSCC_encoder(config).cuda()
    decoder = JSCC_decoder(config).cuda()
    channel = Channel(config)

    if config.CHANNEL.TRAIN_SNR != config.CHANNEL.TRAIN_SINR:
        infcode = JSCC_infencoder(config).cuda()
        infcode_path = config.MODEL.INFCODE_PATH
        infcode.load_state_dict(torch.load(infcode_path))
        infcode.eval()

    train_loader, _, inf_loader, _ = get_loader(config)

    cur_lr = config.LEARNING_RATE
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=cur_lr)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=cur_lr)
    encoder.train()
    decoder.train()

    criterion = loss_matrix(config.TRAIN.LOSS)
    performance = eval_matrix(config.TRAIN.EVAL_LOSS)

    seed_torch()

    losses, eval_matrics, cmptime = [AverageMeter() for _ in range(3)]
    matrix = [losses, eval_matrics, cmptime]
    for e in range(config.TRAIN.EPOCHS):
        for element in matrix:
            element.clear()

        for i, ((input, label1), (inf, label2)) in enumerate(tqdm(zip(train_loader, inf_loader))):
            SNR_list = Str_to_FloatList(config.CHANNEL.TRAIN_SNR)
            SNR = random.choice(SNR_list)
            if config.CHANNEL.TRAIN_SNR == config.CHANNEL.TRAIN_SINR:
                SINR = SNR
            else:
                SINR_list = Str_to_FloatList(config.CHANNEL.TRAIN_SINR)
                SINR = random.choice(SINR_list)

            assert SNR >= SINR, "min of SNR list should not smaller than max of SINR list"

            start_time = time.time()
            input = input.cuda()
            inf = inf.cuda()
            upsample_ratio = input.shape[2] / inf.shape[2]
            inf = torch.nn.functional.interpolate(inf, scale_factor=upsample_ratio, mode="nearest")

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
                sigma_square = 1.0 / (10 ** (SINR / 10))
                # koown SINR : SINR for eqa, else SNR for eqa but SINR = SNR
                noisy_y = torch.conj(h) * noisy_y / (torch.abs(h) ** 2 + sigma_square)
            elif config.CHANNEL.TYPE in ["awgn"]:
                pass
            else:
                raise ValueError

            noisy_y = noisy_y / math.sqrt(torch.mean(torch.abs(noisy_y) ** 2))
            noisy_y = torch.cat((torch.real(noisy_y), torch.imag(noisy_y)), dim=2)

            recon_image = decoder(noisy_y)

            loss = criterion(recon_image, input)
            matric = performance(recon_image, input)

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()

            cmptime.update(time.time() - start_time)

            losses.update(loss.item())
            eval_matrics.update(matric)

            if (i + 1) % (config.TRAIN.TRAIN_PRINT_FRE) == 0:
                log = " | ".join(
                    [
                        f"Epoch {e} print {(i + 1) / (config.TRAIN.TRAIN_PRINT_FRE)}",
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})",
                        f"Time_all {cmptime.sum:.2f}",
                        f"CBR {CBR:.4f}",
                        f"{config.TRAIN.EVAL_LOSS} {eval_matrics.val:.4f} ({eval_matrics.avg:.4f})",
                    ]
                )
                logger.info(log)

    if (e + 1) % config.TRAIN.SAVE_FRE == 0:
        save_model(encoder, save_path=config.MODEL.ENCODER_PATH)
        save_model(decoder, save_path=config.MODEL.DECODER_PATH)
        # test()
        # print(1)


def train_inf_JSCC(config, logger):
    seed_torch()
    logger.info(f"start train inf JSCC seqeratly")
    infencoder = JSCC_infencoder(config).cuda()
    infdecoder = JSCC_infdecoder(config).cuda()
    channel = Channel(config)

    _, _, inf_loader, _ = get_loader(config, need_padding=False)
    upsampling_ratio = config.DATA.IMG_SIZE // config.DATA.INF_IMG_SIZE
    print(upsampling_ratio)
    cur_lr = config.LEARNING_RATE
    optimizer_encoder = optim.Adam(infencoder.parameters(), lr=cur_lr)
    optimizer_decoder = optim.Adam(infdecoder.parameters(), lr=cur_lr)
    infencoder.train()
    infdecoder.train()

    criterion = loss_matrix(config.TRAIN.LOSS)
    performance = eval_matrix(config.TRAIN.EVAL_LOSS)

    seed_torch()

    losses, eval_matrics, cmptime = [AverageMeter() for _ in range(3)]
    matrix = [losses, eval_matrics, cmptime]
    for e in range(config.TRAIN.EPOCHS):
        for element in matrix:
            element.clear()

        for i, (inf, label2) in enumerate(tqdm(inf_loader)):
            SNR_list = Str_to_FloatList(config.CHANNEL.TRAIN_SNR)
            SNR = random.choice(SNR_list)

            start_time = time.time()

            inf = inf.cuda()

            inf = torch.nn.functional.interpolate(
                inf, scale_factor=upsampling_ratio, mode="nearest"
            )

            y = infencoder(inf)
            # print(y.shape)
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

            loss = criterion(recon_image, inf)
            matric = performance(recon_image, inf)

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()

            cmptime.update(time.time() - start_time)

            losses.update(loss.item())
            eval_matrics.update(matric)

            if (i + 1) % (config.TRAIN.TRAIN_PRINT_FRE) == 0:
                log = " | ".join(
                    [
                        f"Epoch {e} print {(i + 1) / (config.TRAIN.TRAIN_PRINT_FRE)}",
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})",
                        f"Time_all {cmptime.sum:.2f}",
                        f"CBR {CBR:.4f}",
                        f"{config.TRAIN.EVAL_LOSS} {eval_matrics.val:.4f} ({eval_matrics.avg:.4f})",
                    ]
                )
                logger.info(log)

        if (e + 1) % config.TRAIN.SAVE_FRE == 0:
            save_model(infencoder, save_path=config.MODEL.INFCODE_PATH)
            save_model(infdecoder, save_path=config.MODEL.INFDECODE_PATH)
        # test()
        # print(1)

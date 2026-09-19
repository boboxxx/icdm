import math

import numpy as np

import torch.optim as optim
from tqdm import tqdm
import copy
from Autoencoder.data.datasets import get_loader
from Autoencoder.loss.distortion import *
from Autoencoder import channel, network
from Diffusion import ICDMTrainer, ICDMSampler

# from Diffusion.Autoencoder import AE

from torchvision.utils import save_image
import torch.nn as nn
from utils import *
import diffusers
from PIL import Image
from DiT.models import DiT_models
from DiT import create_diffusion
from utils import requires_grad, update_ema
from utils import *
from Autoencoder.loss.eval_matrix import loss_matrix, eval_matrix
import copy
from Diffusion import sampling
from Diffusion import sde_lib
import time


def train_ICDM(config, logger, train_for="s"):
    seed_torch()

    lr = config.LEARNING_RATE  # DiT,DiC: 1e-4, consistence model:1e-5,4e-4,5e-6, DDPM: 2e-4, 2e-5
    print("current learning rate:", lr)
    logger.info(f"start train ICDM for {train_for} with learning rate {lr}")
    ICDM = DiT_models(
        in_channels=config.MODEL.OUT_CHANS,
        input_size=config.DATA.IMG_SIZE // (2 ** len(config.MODEL.DEPTHS)),
    ).cuda()

    # test_mem_and_comp(config,ICDM)
    pass_channel = channel.Channel(config)

    JSCC_batch_size = config.DATA.TRAIN_BATCH
    config.DATA.TRAIN_BATCH = config.DATA.TRAIN_ICDM_BATCH
    trainLoader, _, Infloader, _ = get_loader(config)

    config.DATA.TRAIN_BATCH = JSCC_batch_size

    if train_for == "s":
        encoder = network.JSCC_encoder(config).cuda()
        encoder_path = config.MODEL.ENCODER_PATH
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.eval()

    elif train_for == "inf":
        upsampling_ratio = config.DATA.IMG_SIZE // config.DATA.INF_IMG_SIZE
        infcode = network.JSCC_infencoder(config).cuda()

        infcode_path = config.MODEL.INFCODE_PATH
        infcode.load_state_dict(torch.load(infcode_path))

        infcode.eval()
        trainLoader = Infloader
    else:
        raise ValueError

    if config.TrainFromCheckpoint.isspace():
        ICDM.load_state_dict(torch.load(config.TrainFromCheckpoint))

    ema = copy.deepcopy(ICDM).cuda()
    requires_grad(ema, False)
    update_ema(ema, ICDM, decay=0)
    ICDM.train()
    ema.eval()
    diffusion = create_diffusion(timestep_respacing="")
    trainer = ICDMTrainer()

    opt = torch.optim.AdamW(ICDM.parameters(), lr=lr, weight_decay=0)

    # cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer, T_max=ICDM_config.epoch, eta_min=0, last_epoch=-1)
    # warmUpScheduler = GradualWarmupScheduler(
    #     optimizer=optimizer, multiplier=ICDM_config.multiplier, warm_epoch=0.1, #ICDM_config.epoch // 10,
    #     after_scheduler=cosineScheduler)

    # start training

    seed_torch()
    losses, eval_matrics, cmptime = [AverageMeter() for _ in range(3)]
    matrix = [losses, eval_matrics, cmptime]
    for e in range(config.TRAIN.ICDM_EPOCHS):
        for element in matrix:
            element.clear()

        for i, (images, labels) in enumerate(tqdm(trainLoader)):
            # train
            start_time = time.time()
            opt.zero_grad()
            x_0 = images.cuda()
            with torch.no_grad():
                if train_for == "inf":

                    x_0 = torch.nn.functional.interpolate(
                        x_0,
                        scale_factor=upsampling_ratio,
                        mode="nearest",
                    )
                    y = infcode(x_0)
                elif train_for == "s":
                    y = encoder(x_0)

            y, pwr = pass_channel.complex_normalize(y, power=1)  # normalize

            loss = trainer(y, ICDM, diffusion, config.CHANNEL.TYPE, h=None, train_for=train_for)
            loss.backward()

            opt.step()
            update_ema(ema, ICDM)

            cmptime.update(time.time() - start_time)

            losses.update(loss.item())
            # print(
            #     config.MODEL.ICDM_S_PATH[:-4] + f"_epoch{e}_lr{lr}" + ".pth",
            # )
            # torch.save(
            #     ICDM.state_dict(),
            #     "./_epoch{e}" + ".pth",
            # )
            # print("save successfully")
            if (i + 1) % (config.TRAIN.TRAIN_PRINT_FRE) == 0:
                log = " | ".join(
                    [
                        f"Epoch {e} print {(i + 1) / (config.TRAIN.TRAIN_PRINT_FRE)}",
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})",
                        f"Time_all {cmptime.sum:.2f}",
                    ]
                )
                logger.info(log)
        if (e + 1) % config.TRAIN.ICDM_SAVE_FRE == 0:
            logger.info("save model")
            if train_for == "s":
                torch.save(
                    ICDM.state_dict(),
                    config.MODEL.ICDM_S_PATH[:-4] + f"_lr{lr}" + ".pth",
                )
            elif train_for == "inf":
                torch.save(
                    ICDM.state_dict(),
                    config.MODEL.ICDM_Z_PATH[:-4] + f"_lr{lr}" + ".pth",
                )
            else:
                raise ValueError

        if (e + 1) % (config.TRAIN.ICDM_TEST_FRE) == 0:
            save_path = f"./history/eval_result/{config.DATA.DATASET}/{config.MODEL.MODEL_NAME}/{config.TRAIN.EVAL_LOSS}/SNR{config.CHANNEL.TRAIN_SNR}_SINR{config.CHANNEL.TRAIN_SINR}"  # 指明用的何种encoder-decoder
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with torch.no_grad():
                Generate_ICDM(
                    config,
                    model=ICDM,
                    save_name=save_path + f"/epoch{e+1}",
                    train_for=train_for,
                )

        # warmUpScheduler.step()
        # if (e + 1) % config.TRAIN.ICDM_SAVE_FRE == 0:
        #     if train_for == "s":
        #         torch.save(
        #             ICDM.state_dict(),
        #             config.MODEL.ICDM_S_PATH[:-4] + f"_epoch{e}" + ".pth",
        #         )
        #     elif train_for == "inf":
        #         torch.save(
        #             ICDM.state_dict(),
        #             config.MODEL.ICDM_Z_PATH[:-4] + f"_epoch{e}" + ".pth",
        #         )
        #     else:
        #         raise ValueError

    if train_for == "s":
        torch.save(ICDM.state_dict(), config.MODEL.ICDM_S_PATH[:-4] + f"_lr{lr}" + ".pth")
    elif train_for == "inf":
        torch.save(ICDM.state_dict(), config.MODEL.ICDM_Z_PATH[:-4] + f"_lr{lr}" + ".pth")
    else:
        raise ValueError


@torch.no_grad()
def eval_JSCC_with_ICDM(config, logger):
    lr = config.LEARNING_RATE
    seed_torch()
    logger.info("start eval JSCC with ICDM, CBR:{}".format(config.MODEL.OUT_CHANS))
    encoder = network.JSCC_encoder(config).cuda()
    infcode = network.JSCC_infencoder(config).cuda()
    decoder = network.JSCC_decoder(config).cuda()
    infdeco = network.JSCC_infdecoder(config).cuda()
    encoder_path = config.MODEL.ENCODER_PATH
    decoder_path = config.MODEL.DECODER_PATH
    infcode_path = config.MODEL.INFCODE_PATH
    infdeco_path = config.MODEL.INFDECODE_PATH

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    infcode.load_state_dict(torch.load(infcode_path))
    infdeco.load_state_dict(torch.load(infdeco_path))
    pass_channel = channel.Channel(config)

    encoder.eval()
    decoder.eval()
    infcode.eval()
    infdeco.eval()

    train_loader, test_loader, _, test_infloader = get_loader(config)
    print(train_loader.dataset.__len__(), test_loader.dataset.__len__())
    ICDM_s = DiT_models(
        in_channels=config.MODEL.OUT_CHANS,
        input_size=config.DATA.IMG_SIZE // (2 ** len(config.MODEL.DEPTHS)),
    ).cuda()

    ICDM_z = DiT_models(
        in_channels=config.MODEL.OUT_CHANS,
        input_size=config.DATA.IMG_SIZE // (2 ** len(config.MODEL.DEPTHS)),
    ).cuda()

    ckpt_s = torch.load(config.MODEL.ICDM_S_PATH)  # [:-3]+f'_epoch299.pt'
    ckpt_z = torch.load(config.MODEL.ICDM_Z_PATH)  # [:-4] + "_iter{}.pth".format(10 * 9 + 9))
    ICDM_z.load_state_dict(ckpt_z)
    ICDM_s.load_state_dict(ckpt_s)

    ICDM_s.eval()
    ICDM_z.eval()
    # test_mem_and_comp(config, ICDM_s)
    # test_mem_and_comp_of_encoder(config, encoder)
    # test_mem_and_comp_of_decoder(config, decoder)
    ICDM_sampler = ICDMSampler(model_s=ICDM_s, model_z=ICDM_z, config=config).cuda()

    # start training
    SNR_list = Str_to_FloatList(config.CHANNEL.EVAL_SNR)
    SINR_list = Str_to_FloatList(config.CHANNEL.EVAL_SINR)

    (
        mse_with,
        mse_without,
        eval_psnr_with,
        eval_psnr_without,
        eval_msssim_with,
        eval_msssim_without,
        eval_lpips_with,
        eval_lpips_without,
        CLIP_with,
        CLIP_without,
        cmptime,
    ) = [AverageMeter() for _ in range(11)]

    matrix = [
        mse_with,
        mse_without,
        eval_psnr_with,
        eval_psnr_without,
        eval_msssim_with,
        eval_msssim_without,
        eval_lpips_with,
        eval_lpips_without,
        CLIP_with,
        CLIP_without,
        cmptime,
    ]
    PSNR = eval_matrix("PSNR")
    MSSSIM = eval_matrix("MSSSIM")
    LPIPS = eval_matrix("LPIPS")
    CLIP = eval_matrix("CLIP")
    seed_torch()

    for SNR in SNR_list:  # range(0, 75):  # SNR_list:

        # ckpt_z = torch.load(config.MODEL.ICDM_Z_PATH[:-4] + "_iter{}.pth".format(100 + 100 * SNR))
        # ICDM_z.load_state_dict(ckpt_z)
        # SNR = 20
        for SINR in SINR_list:
            assert SNR >= SINR, "min of SNR list should not smaller than max of SINR list"

            for element in matrix:
                element.clear()
            # 这是AWGM,C8的参数
            if config.CHANNEL.TYPE in ["awgn"]:
                Lambda_dict = {
                    "-7": 0.3,
                    "-4": 1,
                    "0": 1.4,
                    "4": 3,
                    "7": 4.5,
                    "10": 5.5,
                    "20": 3,
                }  # Rayleigh 0 dB 0.18 0.43-0.28 with 1.8 dB
                Beta_dict = {"-7": 0.1, "-4": 0.7, "0": 1, "4": 1.0, "7": 1.0, "10": 0.5, "20": 2}
            elif config.CHANNEL.TYPE in ["rayleigh"]:
                # 这是Rayleigh,C8的参数
                Lambda_dict = {
                    # "-10": 0.3,
                    "-7": 0.3,
                    "-4": 0.5,
                    "0": 1,
                    "4": 1.8,
                    "7": 2.2,
                    "10": 2.5,
                }  # Rayleigh 0 dB 0.18 0.43-0.28 with 1.8 dB
                Beta_dict = {
                    # "-10": 0.1,
                    "-7": 0.1,
                    "-4": 0.1,
                    "0": 0.3,
                    "4": 1.0,
                    "7": 1,
                    "10": 1,
                }
            else:
                raise ValueError("invalid channel type")
            Lambda = Lambda_dict[str(int(SINR))]
            Beta = Beta_dict[str(int(SINR))]
            seed_torch()
            for i, ((images, labels), (inf, labels)) in enumerate(
                tqdm(zip(test_loader, test_infloader))
            ):

                # if i < 13:
                #     continue
                # if i > 13:  # 15 1000
                #     break
                x_0 = images.cuda()

                inf = inf.cuda()
                upsample_ratio = x_0.shape[2] // inf.shape[2]
                inf = torch.nn.functional.interpolate(
                    inf, scale_factor=upsample_ratio, mode="nearest"
                )

                y = encoder(x_0)

                y_inf = infcode(inf.cuda())
                #
                y_0 = y

                y, pwr, pwr_inf, h = pass_channel.inf_forward(y, y_inf, SNR, SINR)
                # print(h.shape)
                if config.CHANNEL.TYPE in ["awgn"]:
                    y_eqa = torch.cat((torch.real(y), torch.imag(y)), dim=2)
                    mse1 = torch.nn.MSELoss()(
                        y_eqa * math.sqrt(2),
                        y_0 * math.sqrt(2) / torch.sqrt(pwr),
                    )
                elif config.CHANNEL.TYPE in ["rayleigh"]:
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    y_eqa = y * torch.conj(h) / (torch.abs(h) ** 2 + sigma_square * 2)
                    y_eqa = torch.cat((torch.real(y_eqa), torch.imag(y_eqa)), dim=2)
                    mse1 = torch.nn.MSELoss()(
                        y_eqa * math.sqrt(2),
                        y_0 * math.sqrt(2) / torch.sqrt(pwr),
                    )
                else:
                    raise ValueError("invalid channel type")

                ########## no ICDM for comparison #############
                y_decode = y_eqa / math.sqrt(2 * torch.mean(torch.abs(y_eqa) ** 2))
                x_decode = decoder(y_decode)
                ############ ----------------------##############

                feature_hat, n = ICDM_sampler.SIC_sampling(
                    SNR=SNR,
                    SINR=SINR,
                    deg_feature=y_eqa,
                    h=h,
                    Lambda=Lambda,
                    Beta=Beta,
                    return_intermediate=False,
                )
                # print(len(n))

                feature_hat = feature_hat / math.sqrt(2 * torch.mean(torch.abs(feature_hat) ** 2))
                mse2 = torch.nn.MSELoss()(
                    feature_hat * math.sqrt(2),
                    y_0 * math.sqrt(2) / torch.sqrt(pwr),
                )
                start_time = time.time()
                x_0_hat = decoder(feature_hat)
                t = time.time() - start_time
                print("ICDM time:", t)
                eval_matric_list = [
                    mse2.item(),
                    mse1.item(),
                    PSNR(x_0, x_0_hat.clamp(0.0, 1.0)),
                    PSNR(x_0, x_decode.clamp(0.0, 1.0)),
                    MSSSIM(x_0, x_0_hat.clamp(0.0, 1.0)),
                    MSSSIM(x_0, x_decode.clamp(0.0, 1.0)),
                    LPIPS(x_0, x_0_hat),
                    LPIPS(x_0, x_decode),
                    CLIP(x_0, x_0_hat),
                    CLIP(x_0, x_decode),
                    t,
                ]

                for element, eval_matric in zip(matrix, eval_matric_list):
                    element.update(eval_matric)
                # for t, f in enumerate(n):

                #     f = f / math.sqrt(2 * torch.mean(torch.abs(f) ** 2))
                #     images = decoder(f)
                #     save_image(images, "SIC_{}.png".format(t), nrow=1)

                # save_image(x_0_hat, "SIC_{}.png".format(SINR, config.DATA.DATASET), nrow=5)
                # save_image(x_decode, "noSIC_{}.png".format(SINR, config.DATA.DATASET), nrow=5)
                # save_image(x_0, "origin.png".format(SINR, config.DATA.DATASET), nrow=5)
                log = " | ".join(
                    [
                        f"SNR {SNR} SINR {SINR}",
                        f"t {cmptime.val:.4f}",
                        f"mse_with {mse_with.val:.4f}",
                        f"mse_without {mse_without.val:.4f}",
                        f"PSNR_with {eval_psnr_with.val:.4f}",
                        f"PSNR_without {eval_psnr_without.val:.4f}",
                        # f"MSSSIM_with {eval_msssim_with.val:.4f}",
                        # f"MSSSIM_without {eval_msssim_without.val:.4f}",
                        f"LPIPS_with {eval_lpips_with.val:.4f}({eval_lpips_with.avg:.4f})",
                        f"LPIPS_without {eval_lpips_without.val:.4f}({eval_lpips_without.avg:.4f})",
                        f"CLIP_with {CLIP_with.val:.4f}({CLIP_with.avg:.4f})",
                        f"CLIP_without {CLIP_without.val:.4f}({CLIP_without.avg:.4f})",
                    ]
                )
                logger.info(log)
                if i > 100:  # 15 1000
                    break
            logger.info(i)
            for element in matrix:
                element.append(element.avg)

        if len(SINR_list) > 1:
            logger.info(
                f"SNR {SNR} SINR {SINR_list} MSE_with {mse_with.list}\n MSE_without {mse_without.list}\n PSNR_with {eval_psnr_with.list}\n PSNR_without {eval_psnr_without.list}\n MSSSIM_with {eval_msssim_with.list}\n MSSSIM_without {eval_msssim_without.list}\n LPIPS_with {eval_lpips_with.list}\n LPIPS_without {eval_lpips_without.list}\n CLIP_with {CLIP_with.list}\n CLIP_without {CLIP_without.list}\n"
            )
            for element in matrix:
                element.clear_list()
    if len(SNR_list) <= 1:
        logger.info(
            f"SNR {SNR_list} SINR {SINR} MSE_with {mse_with.list}\n MSE_without {mse_without.list}\n PSNR_with {eval_psnr_with.list}\n PSNR_without {eval_psnr_without.list}\n MSSSIM_with {eval_msssim_with.list}\n MSSSIM_without {eval_msssim_without.list}\n LPIPS_with {eval_lpips_with.list}\n LPIPS_without {eval_lpips_without.list}\n CLIP_with {CLIP_with.list}\n CLIP_without {CLIP_without.list}\n"
        )
        for element in matrix:
            element.clear_list()


@torch.no_grad()
def Generate_ICDM(config, model=None, save_name="uncond_img", train_for="s"):
    # seed_torch(10)
    if train_for == "s":
        decoder = network.JSCC_decoder(config).cuda()
        decoder_path = config.MODEL.DECODER_PATH

    elif train_for == "inf":
        decoder = network.JSCC_infdecoder(config).cuda()
        decoder_path = config.MODEL.INFDECODE_PATH
    else:
        raise ValueError

    decoder.load_state_dict(torch.load(decoder_path))
    decoder.eval()
    if model is None:
        if train_for == "s":

            ckpt = torch.load(config.MODEL.ICDM_S_PATH)
        elif train_for == "inf":

            ckpt = torch.load(config.MODEL.ICDM_Z_PATH)

        DM = DiT_models(
            in_channels=config.MODEL.OUT_CHANS,
            input_size=config.DATA.IMG_SIZE // (2 ** len(config.MODEL.DEPTHS)),
        ).cuda()
        DM.load_state_dict(ckpt)
    else:
        DM = model
    # DM.eval()

    DM_sampler = ICDMSampler(model_s=DM, model_z=DM, config=config).cuda()
    samples = DM_sampler.rand_generate()
    x_0_hat = decoder(samples)
    save_image(x_0_hat, f"{save_name}-{train_for}.png", nrow=5)


@torch.no_grad()
def test_mem_and_comp(config, ICDM):
    from torch_operation_counter import OperationsCounterMode

    network = ICDM.cuda()
    input = torch.randn(1, config.MODEL.OUT_CHANS, 128 // 8, 128 // 8).cuda()

    print(input.shape)
    t = torch.arange(10).cuda()
    with OperationsCounterMode(network) as ops_counter:
        network(input, t)
    # macs,params=profile(network,inputs=(input,))
    # macs, params = clever_format([macs, params], "%.5f")
    print(
        "ICDM:MACs:{}G. Paras:{}M.".format(
            ops_counter.total_operations / 1e9,
            sum([p.numel() for p in [*network.parameters()][:-1]]) / 1e6,
        )
    )


@torch.no_grad()
def test_mem_and_comp_of_encoder(config, encoder):
    from torch_operation_counter import OperationsCounterMode

    network = encoder.cuda()
    input = torch.randn(1, 3, 128, 128).cuda()

    print(input.shape)
    # t = torch.arange(10).cuda()
    with OperationsCounterMode(network) as ops_counter:
        network(input)
    # macs,params=profile(network,inputs=(input,))
    # macs, params = clever_format([macs, params], "%.5f")
    print(
        "Encoder: MACs:{}G. Paras:{}M.".format(
            ops_counter.total_operations / 1e9,
            sum([p.numel() for p in [*network.parameters()][:-1]]) / 1e6,
        )
    )


@torch.no_grad()
def test_mem_and_comp_of_decoder(config, decoder):
    from torch_operation_counter import OperationsCounterMode

    network = decoder.cuda()
    input = torch.randn(1, config.MODEL.OUT_CHANS, 128 // 8, 128 // 8).cuda()

    print(input.shape)
    # t = torch.arange(10).cuda()
    with OperationsCounterMode(network) as ops_counter:
        network(input)
    # macs,params=profile(network,inputs=(input,))
    # macs, params = clever_format([macs, params], "%.5f")
    print(
        "Decoder:MACs:{}G. Paras:{}M.".format(
            ops_counter.total_operations / 1e9,
            sum([p.numel() for p in [*network.parameters()][:-1]]) / 1e6,
        )
    )



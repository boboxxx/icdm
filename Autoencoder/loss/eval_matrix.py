import torch
import numpy as np
from taming.modules.losses.vqperceptual import *
from taming.modules.losses.lpips import LPIPS as lpips
import open_clip
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    """
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    """
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    """
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    """
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out


@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    """
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    """

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs


@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False, eps: float = 1e-8):
    """
    interface of ms-ssim
    :param X: a batch of images, (N,C,H,W)
    :param Y: a batch of images, (N,C,H,W)
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param weights: weights for different levels
    :param use_padding: padding image before conv
    :param eps: use for avoid grad nan.
    :return:
    """
    weights = weights[:, None]

    levels = weights.shape[0]
    vals = []
    for i in range(levels):
        ss, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)

        if i < levels - 1:
            vals.append(cs)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)

    vals = torch.stack(vals, dim=0)
    # Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
    vals = vals.clamp_min(eps)
    # The origin ms-ssim op.
    ms_ssim_val = torch.prod(vals[:-1] ** weights[:-1] * vals[-1:] ** weights[-1:], dim=0)
    # The new ms-ssim op. But I don't know which is best.
    # ms_ssim_val = torch.prod(vals ** weights, dim=0)
    # In this file's image training demo. I feel the old ms-ssim more better. So I keep use old ms-ssim op.
    return ms_ssim_val


class SSIM(torch.jit.ScriptModule):
    __constants__ = ["data_range", "use_padding"]

    def __init__(
        self, window_size=11, window_sigma=1.5, data_range=255.0, channel=3, use_padding=False
    ):
        """
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        """
        super().__init__()
        assert window_size % 2 == 1, "Window size must be odd."
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer("window", window)
        self.data_range = data_range
        self.use_padding = use_padding

    @torch.jit.script_method
    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return r[0]


class MS_SSIM(torch.jit.ScriptModule):
    __constants__ = ["data_range", "use_padding", "eps"]

    def __init__(
        self,
        window_size=11,
        window_sigma=1.5,
        data_range=1.0,
        channel=3,
        use_padding=False,
        weights=None,
        levels=None,
        eps=1e-8,
    ):
        """
        class for ms-ssim
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels
        :param use_padding: padding image before conv
        :param weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        :param levels: number of downsampling
        :param eps: Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
        """
        super().__init__()
        assert window_size % 2 == 1, "Window size must be odd."
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer("window", window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer("weights", weights)

    @torch.jit.script_method
    def forward(self, X, Y):
        return 1 - ms_ssim(
            X,
            Y,
            window=self.window,
            data_range=self.data_range,
            weights=self.weights,
            use_padding=self.use_padding,
            eps=self.eps,
        )


def get_clip_score_images(image_tensor1: torch.Tensor, image_tensor2: torch.Tensor) -> torch.Tensor:
    """
    计算两个 PyTorch 图像张量之间的 CLIP 余弦相似度分数。

    Args:
        image_tensor1 (torch.Tensor): 第一个图像张量。
                                      形状应为 (C, H, W)，其中 C 是通道数 (1 或 3)，
                                      H 是高度，W 是宽度。
                                      数据类型应为 torch.float32。
                                      数值范围应在 [0.0, 1.0] 之间。
        image_tensor2 (torch.Tensor): 第二个图像张量，格式和范围与第一个相同。

    Returns:
        torch.Tensor: 计算得到的余弦相似度（一个标量 tensor）。
                      值越接近 1 表示两个图像在 CLIP 语义空间中越相似。
    """
    # 1. 加载 CLIP 模型和预处理函数
    # 这里使用 open_clip 库加载一个预训练的 CLIP 模型，例如 "ViT-B-32"
    # pretraind='openai' 表示使用 OpenAI 训练的权重
    # print("加载 CLIP 模型和预处理函数...")
    local_path = "/home/wt/code/SIC-JDM/SIC-JDM/taming/open_clip_pytorch_model.bin"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=local_path)
    # print("模型加载完成。")

    # 确定使用的设备 (GPU 如果可用，否则是 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # print(f"使用设备: {device}")

    # 确保输入的 tensor 在正确的设备上
    image_tensor1 = image_tensor1.to(device)
    image_tensor2 = image_tensor2.to(device)

    # 将 tensor (CxHxW, float [0, 1]) 转换为 PIL Image (HxWxC, uint8 [0, 255])
    # 注意：这里假设输入是 [0, 1]，需要乘以 255 转换为 [0, 255] 的 uint8

    # 应用 CLIP 预处理 transforms
    # preprocess 函数会执行 Resize, CenterCrop, ToTensor, Normalize 等步骤
    # ToTensor 会将 PIL Image (HxWxC, uint8 [0, 255]) 转换回 tensor (CxHxW, float [0, 1])
    # Normalize 会应用 CLIP 训练时使用的均值和标准差，将数值范围改变
    CLIP_INPUT_SIZE = 224
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    preprocess_batch = T.Compose(
        [
            T.Resize(CLIP_INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(CLIP_INPUT_SIZE),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )

    # 应用预处理 (直接在整个批次上)
    # T.Resize and T.CenterCrop work on batches of tensors (B, C, H, W)
    # T.Normalize also works on batches
    processed_image1 = preprocess_batch(image_tensor1)
    processed_image2 = preprocess_batch(image_tensor2)
    # print("预处理完成。")
    # print("预处理完成。")

    # 3. 使用 CLIP 模型编码图像获取特征向量
    # print("编码图像获取特征向量...")
    with torch.no_grad():  # 在推理阶段，不需要计算梯度
        image_features1 = model.encode_image(processed_image1)
        image_features2 = model.encode_image(processed_image2)
    # print("编码完成。")

    # 4. 对特征向量进行 L2 归一化
    # CLIP 的特征向量通常是 L2 归一化后的，以便计算余弦相似度（点积即余弦相似度）
    image_features1 = F.normalize(image_features1, dim=-1)
    image_features2 = F.normalize(image_features2, dim=-1)

    # 5. 计算两个特征向量之间的余弦相似度
    clip_score = F.cosine_similarity(image_features1, image_features2, dim=-1)

    return clip_score


class loss_matrix(torch.nn.Module):
    def __init__(self, type):
        super(loss_matrix, self).__init__()
        # self.config=config
        self.Cal_lpips = lpips().eval().cuda()
        _loss_dict = dict(
            PSNR=self.MSE_loss, MSSSIM=self.MSSSIM_loss, LPIPS=self.LPIPS_loss, CLIP=self.CLIPscore
        )
        self.loss = _loss_dict.get(type, None)

    def MSSSIM_loss(self, x, y):
        CalcuSSIM = MS_SSIM(data_range=1.0, levels=4, channel=3).cuda()
        rec_loss = CalcuSSIM(x, y).mean() * x.numel() / x.shape[0]

        return rec_loss

    def MSE_loss(self, x, y):

        rec_loss = torch.nn.functional.mse_loss(x, y, reduction="sum") / x.shape[0]

        return rec_loss

    def LPIPS_loss(self, x, y):

        rec_loss = self.Cal_lpips.forward(x, y).mean() * x.numel() / x.shape[0]
        # print(rec_loss)
        return rec_loss

    def CLIPscore(self, x, y):
        semantic_loss = get_clip_score_images(x, y)
        return semantic_loss

    def forward(self, recon, input):

        return self.loss(recon, input)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight


class eval_matrix(torch.nn.Module):
    def __init__(self, type):
        super(eval_matrix, self).__init__()
        self.Cal_lpips = lpips().eval().cuda()
        _loss_dict = dict(
            PSNR=self.psnr, MSSSIM=self.msssim, LPIPS=self.LPIPS_loss, CLIP=self.CLIPscore
        )
        self.loss = _loss_dict.get(type, None)

    def psnr(self, x, y):
        # print(x.shape,y.shape)
        mse = torch.nn.MSELoss()(x.clamp(0.0, 1.0) * 255.0, y.clamp(0.0, 1.0) * 255.0)
        # loss=torch.nn.MSELoss(x.clamp(0.,1.), y.clamp(0.,1.))
        psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10)).item()
        return psnr

    def msssim(self, x, y):
        CalMSSSIM = MS_SSIM(data_range=1.0, levels=4, channel=3).cuda()
        msssim = 1 - CalMSSSIM(x, y).mean().item()
        return msssim

    def LPIPS_loss(self, x, y):

        rec_loss = self.Cal_lpips(x, y).mean().item()
        return rec_loss

    def forward(self, x, y):
        return self.loss(x, y)

    def CLIPscore(self, x, y):
        semantic_loss = get_clip_score_images(x, y).mean().item()
        return semantic_loss

# SC-ICDM 第一阶段：代码、运行方式与边界

## 当前状态（2026-09-15）

- sheng：`sheng@100.94.183.27`，主机 `DESKTOP-UGDDO8T`，RTX 4090。
- 远端目录：`/home/sheng/ICDM_SC_20260915`。
- 远端分支：`codex/sc-icdm-phase1`；原始基线提交：`e6b7f8d5eb8161e345dc5bebd7bcf0f5fcc51925`。
- 复用环境：`/home/sheng/anaconda3/envs/cv/bin/python`，PyTorch 2.5.0+cu118。
- 已实现 A0–A3 与独立评估入口；数值与 CUDA 集成检查的日志见 `runs/phase1_checks/tests.log`。
- **尚未运行训练权重支持的图像性能实验；不能从集成测试推断恢复质量或完整模型时延。**

## 权重核查

官方 [仓库](https://github.com/Wireless3C-SJTU/ICDM) 当前提交与本地基线一致。GitHub API 的完整文件树中未发现 `.pth/.pt/.ckpt/.safetensors`；[Releases](https://github.com/Wireless3C-SJTU/ICDM/releases) 列表为空。README 未提供权重下载链接。

[Issue #1](https://github.com/Wireless3C-SJTU/ICDM/issues/1) 询问缺失的主程序入口，截至核查时没有回复。根目录 `utils.py` 也缺失，所以新推理入口隔离了训练模块的自动导入。

在 sheng 的用户主目录（有限深度）、Weibo 的 Windows 桌面/文档和 D 盘代码目录内，未找到匹配 ICDM/INFCODE/ENCODER_SNR/DECODER_SNR 文件名的权重。这不证明磁盘其他位置或重命名文件中不存在权重。

接收性能实验需同一配置下匹配的 encoder、decoder、ICDM-S、ICDM-Z；若从干扰图像生成潜变量，还需 infencoder。此入口不需要 infdecoder。路径和模型结构必须明确匹配，加载使用 strict=True；不会以随机权重继续运行。

## 实验定义

| 模式 | 接收端输入 | 干扰幅度 | Lambda/Beta |
|---|---|---|---|
| A0 | 真实整帧 SINR | 原公式 | 原整数键查表 |
| A1 | 真实整帧 SINR | sqrt(10^(-SINR/10) − 10^(-SNR/10)) | 与 A0 相同 |
| A2 | 原始复数接收信号、CSI、SNR | 整帧能量估计 | 由估计 SINR 插值并截断至表端点 |
| A3 | 原始复数接收信号、CSI、SNR、块长 | 固定分块能量估计 | 由各块估计 SINR 插值并截断 |

所有模式固定 40 steps、order=3、noise_prediction。A2/A3 的接收方法没有真实 SINR、干净信号或真实块功率参数。A3 仅初始化功率，不是交替更新幅度的 A4，也不是后验矩校准的 A5。

块按每帧复数张量的 channel → height → width 展平顺序划分，末块可以不足指定长度；同一幅度重复至实部和虚部。干扰先验全局归一化，不对每块独立归一化。能量估计先取块平均，再减去已知期望信号能量与噪声功率，最后截断为非负数。已知 CSI 下局部期望信号功率为 |h|²；其单位功率/独立假设在短块可能不准确。

三种仿真：平稳、交替功率、间歇突发。非平稳块功率按块长加权，保持相同名义整帧 SINR。非平稳情形下 A0/A1 只知道整帧 SINR，**不是知道所有块功率的 oracle 上界**。

## 尚未扩展的物理一致性问题

A1 只修正幅度公式。Rayleigh 均衡器原有的 2*N0 分母与采样引导的 N0 分母差异仍保留，便于单独考察幅度修正。因此不能把 A1 宣称为整条链路完全物理一致的基线。

原 Lambda/Beta 表在代码中标为 C8，未针对新幅度公式重新调参。A0/A1 的对照是固定参数消融，不代表各自最优性能。A2/A3 的估计 SINR 超出表范围时，仅引导参数截断到端点，幅度继续使用估计值。

## 准备权重后运行

复制 `configs/phase1.example.json` 并填写实际路径、结构和数据尺寸。示例参数不是对现有权重结构的确认。对于包含 `ema` 等外层键的 checkpoint，在对应权重项中显式设置 `"state_key": "ema"`。

```bash
cd /home/sheng/ICDM_SC_20260915
PYTHONDONTWRITEBYTECODE=1 /home/sheng/anaconda3/envs/cv/bin/python -m scripts.run_phase1 \
  --manifest configs/phase1.json --output runs/phase1_real --preflight

PYTHONDONTWRITEBYTECODE=1 /home/sheng/anaconda3/envs/cv/bin/python -m scripts.run_phase1 \
  --manifest configs/phase1.json --output runs/phase1_real
```

测试输入为本地图像目录；目标图像中心裁剪至 image_size；干扰图像中心裁剪至 interference_image_size 后最近邻上采样。不会下载数据。平稳场景沿用原信道生成函数。每组四模式共享观测、CSI、初始化种子；每模式首次执行一次预热，预热不计入结果。

输出：`metadata.json`（配置、源码/权重 SHA256、状态）、`records.jsonl`（逐图 PSNR/MSE、估计功率、时延及事后记录的仿真真值）。每条记录包含实际图像路径。当前实现 PSNR/MSE，**未实现 LPIPS/MSSSIM/CLIP，也不构成论文全指标复现**。

GPU 同步后分别记录采样+估计/均衡时间、归一化+解码时间，以及二者合计。数据读入、编码器和信道仿真不属于接收端计时。测试样本上不调参数。

## 检查命令

```bash
PYTHONDONTWRITEBYTECODE=1 ICDM_TEST_DEVICE=cuda:0 \
  /home/sheng/anaconda3/envs/cv/bin/python -m unittest discover -s tests -v
```

检查包括原提交输出一致性、幅度功率关系、零干扰、复数实虚布局、末块、逐帧能量估计及两类信道的盲采样。采样检查使用明确的测试替身，不加载或冒充训练模型。

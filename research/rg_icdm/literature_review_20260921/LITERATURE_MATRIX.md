# 文献矩阵：语义通信、生成式恢复与接收决策

检索日：2026-09-21。共 30 篇独立文献。

**F：12 篇，阅读了全文中的关键方法、理论或实验部分；不表示逐页读完附录或重新证明全部定理。A：18 篇，摘要与元数据筛选，尚不能据此确认实现细节或论文性能主张。** 所有观点均用于本项目研究决策，不把论文自行声称的最优性当成独立验证结果。年份优先注明所读预印本/版本；未核验最终出版元数据的论文不猜测卷期或 DOI。

## 直接相关的通信方法

| ID / 深度 | 论文与原始来源 | 年份 / 作者 | 已读位置与核心信息 | 对本项目的作用及边界 |
|---|---|---|---|---|
| 01 / A | [Deep Joint Source-Channel Coding for Wireless Image Transmission](https://arxiv.org/abs/1809.01733) | 2018 预印本；Bourtsoulatze et al. | 摘要；联合学习图像到信道符号的映射 | 奠定直接解码基线；不等于当前 Swin 编码器的独立复现 |
| 02 / A | [SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding](https://arxiv.org/html/2308.09361v2) | 2023 起；Yang et al. | 摘要与架构概览；Transformer 编码及信道/码率适配 | 当前工程的表示基础；固定权重实验不能声称复现完整自适应系统 |
| 03 / A | [Generative Joint Source-Channel Coding for Semantic Image Transmission](https://arxiv.org/abs/2211.13772) | 2022 预印本；Erdemir et al. | 摘要；InverseJSCC 与端到端 GenerativeJSCC | 生成先验改善感知已有先例；StyleGAN 与现有潜变量先验不是相同资产 |
| 04 / F | [CDDM: Channel Denoising Diffusion Models for Wireless Semantic Communications](https://arxiv.org/html/2309.08895) | 2023 预印本；Wu et al. | §III-B 算法 2、§III-C、§IV-B 算法 3；噪声匹配初始化与确定性去噪，完整系统含解码器重训练 | 必须补盲总扰动功率版本；现有 DiT/固定解码器/重采样版本仅为 receiver port |
| 05 / F | [ICDM: Interference Cancellation Diffusion Models for Wireless Semantic Communications](https://arxiv.org/html/2505.19983v1) | 2025 v1；Wu et al. | §II 系统与假设、§IV 实验；目标和干扰双先验、联合后验恢复 | 最接近的母方法；必须披露干扰先验和功率信息。理论假设不可直接当作当前神经先验保证 |
| 06 / F | [DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling](https://arxiv.org/html/2406.07390) | 2024 起；Wang et al. | §III-B/C/D，算法 1/2；测量约束、confirming constraint、自适应初始化和盲信道恢复 | 约束“一致性反馈、自适应采样、盲恢复”的创新声明；不是与当前 DiT 同算力的现成替换 |
| 07 / F | [Deep Joint Source-Channel Coding with Iterative Source Error Correction](https://proceedings.mlr.press/v206/lee23c.html) / [全文](https://arxiv.org/html/2302.09174) | AISTATS 2023；Lee, Hu, Kim | §3.1–3.3；编码—解码循环似然、bias-free denoiser、噪声失配下步长/正则调节 | 不能把普通循环修正冒充 ISEC；需匹配其先验训练与通信预算 |
| 08 / A | [High Perceptual Quality Wireless Image Delivery with Denoising Diffusion Models](https://arxiv.org/html/2309.15889v2) | 2023 起；Yilmaz et al. | 摘要；range/null-space 与扩散恢复 | 感知恢复基线家族；失真下降与感知收益应分别测量 |
| 09 / F | [SING: Semantic Image Communications using Null-Space and INN-Guided Diffusion Models](https://arxiv.org/html/2503.12484v1) | 2025；Chen et al. | §III、§IV-A/B、算法 1；线性代理与 INN 两类前向模型 | 盲图像恢复已有方法；SING-Zero 的代理算子一致性不等于原始实例恢复保证 |
| 10 / F | [Enabling Training-Free Semantic Communication Systems with Generative Diffusion Models](https://arxiv.org/html/2505.01209v1) | 2025；Tang et al. | §III 与算法 1；冻结 Stable Diffusion 组件，发送前 DDIM inversion、接收端对齐与采样 | “training-free”已经不是新点；本项目若用标注校准需说明训练/校准边界 |
| 11 / F | [SecDiff: Diffusion-Aided Secure Deep Joint Source-Channel Coding Against Adversarial Attacks](https://arxiv.org/html/2511.01466v1) | 2025；Zhao et al. | §IV-D/E、算法与实验对照描述；功率掩码、伪逆引导、EM 算子更新 | 功率检测+扩散+EM 已进入通信；其导频/OFDM/攻击假设与当前加性干扰不同 |
| 12 / A | [Latent Feature-Guided Conditional Diffusion for Generative Image Semantic Communication](https://arxiv.org/abs/2504.21577) | 2025 v2；Chen et al. | 摘要；ROI 加权、可变码率和条件扩散 | 语义重要性引导不是空白；需要 ROI 信息与训练支持。使用 v2 修订标题 |
| 13 / A | [Generative Semantic Communication: Diffusion Models Beyond Bit Recovery](https://arxiv.org/abs/2306.04321) | 2023/2026 v2；Grassucci, Barbarossa, Comminiello | 摘要；发送压缩语义后生成场景，关注对象/位置/深度 | 帮助区分任务语义与像素重建；不与固定潜变量传输预算直接排榜 |
| 14 / A | [Generative AI Meets 6G and Beyond: Diffusion Models for Semantic Communications](https://arxiv.org/abs/2511.08416) | 2025/2026 v3；Qin et al.；页面注明 COMST accepted | 摘要；条件、效率与泛化三个方向及逆问题视角 | 用作领域地图和引用追踪；综述本身不证明项目创新或实验效果 |
| 15 / A | [Wireless Hallucination in Generative AI-enabled Communications: Concepts, Issues, and Solutions](https://arxiv.org/abs/2503.06149) | 2025；Wang et al. | 摘要；生成式无线系统幻觉问题 | 概念背景；不能凭 PSNR 下降判定当前图像存在语义幻觉 |
| 16 / A | [FAST: Flexible and adaptive semantic transmission for resource-constrained multi-user generative semantic communication](https://pure.qub.ac.uk/en/publications/fast-flexible-and-adaptive-semantic-transmission-for-resource-con/) | 2026；Wang et al.；作者机构记录 | 摘要；顺序语义传输、条件去噪与资源适配 | 多用户资源调度已有竞争；超出本项目近期固定链路范围 |

## 生成式逆问题与统计依据

| ID / 深度 | 论文与原始来源 | 年份 / 作者 | 已读位置与核心信息 | 对本项目的作用及边界 |
|---|---|---|---|---|
| 17 / A | [Denoising Diffusion Restoration Models](https://arxiv.org/abs/2201.11793) | 2022；Kawar et al. | 摘要；预训练先验处理带噪线性逆问题 | G 类单先验接收的思想依据；依赖算子与噪声模型 |
| 18 / A | [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687) | 2022 起；Chung et al. | 摘要与论文 PDF 检索段；带噪逆问题的近似引导 | 对似然近似必须保留误差边界；不是精确后验采样的通用保证 |
| 19 / A | [Parallel Diffusion Models of Operator and Image for Blind Inverse Problems](https://arxiv.org/abs/2211.10656) | 2022 / CVPR 2023；Chung et al. | 摘要；同时对图像和算子使用扩散先验 | 双模型盲逆问题已有先例；算子先验训练成本不可省略 |
| 20 / F | [GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration](https://proceedings.mlr.press/v202/murata23a.html) / [PDF](https://proceedings.mlr.press/v202/murata23a/murata23a.pdf) | ICML 2023；Murata et al. | §2、§3.3 与式 17 的检索全文段；交替估计算子与信号，可用简单算子先验 | 参数反馈已有严格框架；不需要为本项目每个标量再训练大型先验 |
| 21 / F | [Fast Diffusion EM: a diffusion model for blind inverse problems with application to deconvolution](https://arxiv.org/abs/2309.00287) | 2023 / WACV 2024；Laroche, Almansa, Coupete | CVF 全文检索返回 §3 式 31–33、§4.1；近似 E 步与算子 M 步 | 否定“EM 更新本身新颖”；不能将去卷积实验的优势直接外推到语义干扰 |
| 22 / F | [Tweedie Moment Projected Diffusions for Inverse Problems](https://arxiv.org/html/2310.06721) | 2023 起；Boys et al. | §3.1–3.4、§4 开头；一二阶矩、Gaussian projection、Jacobian 成本 | 支持审查矩近似；精确 Tweedie 恒等式不等于近似网络的真实后验协方差 |
| 23 / A | [The Perception-Distortion Tradeoff](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html) | CVPR 2018；Blau, Michaeli | 摘要；分布感知质量与失真的权衡 | 不能把平均 PSNR/MSE 的聚合差异误称此权衡；LPIPS 也不等于完整语义正确性 |
| 24 / F | [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/html/2110.01052) | 2021 起；Angelopoulos et al. | §1.1、§2.1–2.5、§3.2；有限策略检验、FWER、选择风险 | 可用于预先固定的策略族和独立校准；本项目不能把现成定理当原创，或忽略同图重复 |
| 25 / A | [Conformal Risk Control](https://arxiv.org/abs/2208.02814) | 2022 起；Angelopoulos et al. | 摘要；单调损失的期望风险控制 | 不能直接把非单调接收机切换塞入原始条件 |
| 26 / A | [Conformal Risk Control for Non-Monotonic Losses](https://arxiv.org/abs/2602.20151) | 2026；Angelopoulos | 摘要；多维非单调选择，界依赖算法稳定性 | 说明该理论方向仍在推进；未核验全部条件，当前先采用简单有限族检验 |

## 进一步压缩创新空间的近期工作

| ID / 深度 | 论文与原始来源 | 年份 / 作者 | 已读位置与核心信息 | 对本项目的作用及边界 |
|---|---|---|---|---|
| 27 / A | [Asymmetric Diffusion Based Channel-Adaptive Secure Wireless Semantic Communications](https://arxiv.org/abs/2310.19439) | 2023；Ren et al. | 摘要；DiffuSeC，DRL 选择扩散步数 | “信道自适应步数”不能作为首次提出；DRL 不是本项目短期必要组件 |
| 28 / A | [Channel-Aware Preemptive Scheduling for Semantic Communication with Truncated Diffusion and Path Compensation](https://arxiv.org/html/2604.11849) | 2026；Liang, Li | 摘要与结构筛选；信道驱动截断、路径补偿 | 把通信状态与生成时延联合考虑已有工作；本项目聚焦接收模型选择 |
| 29 / F | [Diffusion-Aided Bandwidth-Efficient Semantic Communication with Adaptive Requests](https://arxiv.org/html/2510.26442) | 2025/2026 v4；Wang et al. | §II-B/C 算法 1、§III-A；文本一致性、有/无引导选择、追加块和停止 | 不能声称首次接收端选择/早停；其文本、控制元数据、反馈与计算代价需和本方案区别 |
| 30 / A | [Latent Diffusion Model Based Denoising Receiver for 6G Semantic Communication: From Stochastic Differential Theory to Application](https://arxiv.org/abs/2506.05710) | 2025 v3；Wang, Jia, Cheng | 摘要；SNR—时间映射与特征尺度适配 | 与“免训练、尺度校准、初始化”直接相邻；“optimal”是作者主张，未完成独立验证 |

## 引文核验与复现优先级

正式写稿首先引用 04–07、09、11、24、29：它们最直接限定论文的新颖性。01–03、08、13–14 是背景；17–22 是数学来源；25–26 仅在真的采用相应统计方法时进入主文。五页论文不必把全部 30 篇强行塞入引用。

以相同表示与先验验证机制时，优先实现 CDDM receiver port、ICDM 原始实现、A4+同 gate 和单先验异方差高斯控制。完整 DiffCom/SING/ISEC 系统若使用不同编码率、模型容量或训练集，应另列系统级比较，不能直接拿原论文分数比较。未实现的相关方法必须诚实标为文献对比，不能伪造实验行。

CDDM 的短会议版与长文作为同一工作家族处理，未重复计数。LRISC 使用 v2 改名后的标题。ICDM 所读正文是 2025 v1；本地既有协议记录其后续出版 DOI，但本矩阵不把 v1 文字自动当成最终出版版本。引文引用数、h-index 和所谓影响力排名没有可靠查询，因此未填写。

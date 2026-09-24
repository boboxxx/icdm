# VTC论文写法核对与本稿重排

核对日期：2026-09-23。阅读对象为公开全文，不把写法观察等同于所有VTC论文的统一规定，也不复制原文句式。

## 阅读来源

1. Weiran Jiang, Wei Chen, Bo Ai, “Deep Joint Source Channel Coding With Attention Modules Over MIMO Channels,” VTC2024-Spring，5页。官方全文：https://ieeevtc.org/vtc2024spring/DATA/PID2024001863.pdf 。重点看第1页引言、第2页系统模型、图1–3标题。文献融入引言；先给源图像、编码映射、带宽和功率约束，再给信道关系与接收目标；方法单列下一节。
2. Shunpu Tang et al., “Contrastive Learning based Semantic Communication for Wireless Image Transmission,” VTC2023-Fall，作者公开稿5页：https://arxiv.org/pdf/2304.09438 。会议身份核对：https://kclpure.kcl.ac.uk/portal/en/publications/contrastive-learning-based-semantic-communication-for-wireless-im/ 。重点看引言中背景、已有方法局限、设计动机的衔接，以及系统模型的编码—归一化—信道—解码顺序。图题简短交代内容或测试条件。
3. Kaiwen Yu, Qi He, Gang Wu, “Partial Sampling-based Semantic Communications for Internet of Things,” VTC2024-Spring，官方全文6页：https://ieeevtc.org/vtc2024spring/DATA/PID2024002126.pdf 。重点看System Model and Problem Description：先界定发送端、接收端与可用信息，再说明研究目标；不能先堆算法细节再补系统假设。

## 本稿采用的组织方式

- 第I节Introduction：JSCC背景与必要文献 → 去噪/分离的矛盾 → 盲选择问题 → 方法概览 → 三项贡献 → 评价边界和章节安排。引言延伸至第一页底部，System Model从第二页开始；不单设Related Work。
- 第II节System Model and Problem Formulation：图像与两个发送端 → 复码字及帧功率约束 → 分块干扰与接收端已知/未知信息 → 共用解码器和选择目标。新增公式只是把已有实现显式写出，不修改模型或实验。
- 第III节Blind Disturbance-Model Selection：候选扰动模型 → 两个接收机 → 选择规则与开发集拟合。
- 第IV节Experimental Evaluation：实验设置 → 冻结规则和配对比较 → 推理成本与退化 → 失配测试。保留第三方DeepJSCC/官方MambaJSCC来源区别、单训练种子和固定40-epoch预算，以及跨骨干接收张量不同的限制。
- 第V节Discussion and Conclusion：核心发现、逐帧风险、部署边界。

## 图表标题

Table I: Reconstruction quality and receiver cost on the confirmation set.

Figure 1: Mean PSNR versus nominal SINR over the three interference profiles.

Figure 2: Empirical CDF of per-frame PSNR change relative to calibrated joint sampling.

Table II: Mean PSNR under block mismatch and no interference.

Figure 3: Reconstructed images for six prespecified confirmation examples.

原图题中的统计单位、同门控计时、实虚线区别、损失阈值、预先指定案例等信息移入对应正文，没有因缩短标题而删除。最终维持IEEE标准字号与页边距，未用负间距或缩放正文凑页数。


## 章节导语与结论修订（2026-09-24）

第II、III、IV节分别用“In this section, …”说明本节的系统建模、候选接收机与选择规则、实验比较内容。第V节改为Conclusion，以“In this paper, we proposed …”起笔，依次概括方法、独立确认的PSNR增益与计算节省、适用范围和未来工作。压缩了重复解释并调整浮动图表位置；仍为标准IEEE双栏5页，Introduction在第一页结束，系统模型从第二页开始。五页均已重新视觉检查，编译无未解析引用或横向溢出；图表标题维持简短单句。实验数据、权重、阈值和确认结果未变。


## Introduction 论证链修订（2026-09-24）

按用户提供的 Problem → Existing paradigm → Gap → Why the gap matters → Insight → Method → Evidence → Contribution 逻辑重写引言。

1. 问题与评价标准：未知同频干扰下，冻结编解码器需要恢复图像，同时控制接收计算量。
2. 现有范式：Gaussian去噪与联合信号/干扰重建提供两种恢复机制，但接收机仍须决定对当前观测使用哪一种。
3. 核心不足：联合重建利用结构，也必须额外估计干扰潜变量与幅度；估计误差可能同时损害图像和增加计算。
4. Insight：显式干扰建模的收益与估计代价随观测变化，因此应在生成推理前选择模型；接收能量及其块间变化提供可观测依据，其有效性须由验证确定。
5. 方法：图像分组验证选择阈值规则，必要时退回固定接收机；冻结规则后只运行一个分支，不更新神经权重。
6. 证据：独立确认相对Gaussian和joint的平均PSNR增益分别为0.750/1.334 dB，相对joint少34.7%预测器调用；相同门控控制、有害决策和块失配限定结论。
7. 贡献：接收决策问题、低成本推理前选择、配对独立评估形成对应关系，无未经证实的SOTA主张。

仅修改Introduction，摘要及第II节以后的正文与上一版逐字一致；未增加或替换引用，保留完整CDDM/ICDM未复现的范围说明。全文仍为5页，Introduction在第一页结束，第二页以System Model开始。重新编译与五页视觉检查通过，未缩小字体或页边距。


## System Model 链路与问题定义修订（2026-09-24）

按照用户提供的 source → representation → transmission → corruption → diffusion recovery → quality 思路，将第II节改为三个小节：

- Image Representation and Transmission：定义RGB源、JSCC联合表示与信道映射、归一化、带宽比例、分块干扰信道以及接收机已知/未知信息。
- Diffusion-Assisted Image Recovery：把原第III节的两种扰动模型移入系统模型，说明观测提供帧级证据、先验提供分布信息，随后由选定接收机恢复潜变量并交给公共解码器。保留未知幅度估计与混合信号非唯一性的说明。模型使用现有冻结先验和观测初始化/引导，不宣称精确后验采样或新增条件去噪网络。
- Quality, Cost, and Receiver Selection：定义逐图像PSNR、预测器调用数和接收延迟，并加入开发集平均PSNR最大化公式。规则族先由图像分组验证确定，包含固定接收机回退，再拟合规则参数。明确带宽、功率和神经权重固定；成本独立报告，未虚构功率/码率/步数联合优化、端到端时延约束或语义任务指标。

第III节接续具体候选接收机更新与推理前选择规则。Introduction、候选接收机实现说明、选择算法、实验和结论与上一版逐字一致；全部实验数据、冻结阈值与权重未改动。全文仍为IEEE标准双栏5页，第一页Introduction、第二页System Model，保持三个主要章节的“In this section”导语与“In this paper”结论。重新编译、交叉引用和五页视觉检查通过，无横向溢出。

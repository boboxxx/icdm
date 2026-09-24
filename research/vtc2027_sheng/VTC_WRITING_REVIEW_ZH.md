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

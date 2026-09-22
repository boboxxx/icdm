# VTC 论文的章节逻辑、词数与本稿重写

检索与重写日期：2026-09-19。

这次以公开全文为依据，检查了 7 篇已发表 VTC 研究对应的可获取版本。核心篇幅样本为 5 篇 5–6 页全文，另有一篇 2 页短文和一篇 7 页作者稿用于补充观察。样本集中于语义通信、图像传输和干扰处理，是有目的的小样本，不代表 VTC 全体论文的统计分布。

最稳定的结构观察是：核心 5 篇均采用“引言 → 系统模型/问题 → 方法 → 仿真 → 结论”五个主体章节。相关工作融入引言，实验设置放在仿真节开头，没有独立的 Related Work 或 Limitations 主节。这是样本中的写作选择，不是会议强制规定。

## 1. 全文来源与计数口径

| 编号 | 论文与可核实全文 | 所读版本 |
|---|---|---|
| P1 | [Contrastive Learning based Semantic Communication for Wireless Image Transmission](https://arxiv.org/abs/2304.09438)，Tang 等，VTC 2023-Fall；[正式出版 DOI](https://doi.org/10.1109/VTC2023-Fall60731.2023.10333392) | arXiv v1，5 页；正式出版记录为 1–6 页，以下词数仅适用于这份公开作者稿 |
| P2 | [Universal Weighted-Knowledge Bases for Task-Unaware Semantic Communication Systems](https://ieeevtc.org/vtc2024spring/DATA/PID2024002016.pdf)，Jiang 等，VTC 2024-Spring | 官方会议论文集 PDF，5 页 |
| P3 | [Partial Sampling-based Semantic Communications for Internet of Things](https://ieeevtc.org/vtc2024spring/DATA/PID2024002126.pdf)，Yu 等，VTC 2024-Spring | 官方会议论文集 PDF，6 页 |
| P4 | [Spectral Efficiency Maximization for Probabilistic Semantic Communication with Rate Splitting](https://ieeevtc.org/vtc2024spring/DATA/PID2024002149.pdf)，Zhao 等，VTC 2024-Spring | 官方会议论文集 PDF，5 页 |
| P5 | [Efficient Design for NOMA Enabled Integrated Sensing and Semantic Communication](https://ieeevtc.org/vtc2024spring/DATA/PID2024002614.pdf)，Zhao 等，VTC 2024-Spring | 官方会议论文集 PDF，5 页 |
| S1 | [Blind Co-channel Interference Cancellation Using Fast Fourier Convolutions](https://ieeevtc.org/vtc2024spring/DATA/PID2024002600.pdf)，Naseri 等，VTC 2024-Spring；[作者机构记录](https://biblio.ugent.be/publication/01JGX8ERBWDVZF8RHSXTBQNDDG) | 官方 2 页短文，不并入完整论文词数基准 |
| S2 | [Digital Twin Aided RIS Communication: Robust Beamforming and Interference Management](https://arxiv.org/abs/2406.04188)，Alikhani 与 Alkhateeb；[机构出版记录：VTC 2024-Fall](https://asu.elsevierpure.com/en/publications/digital-twin-aided-ris-communication-robust-beamforming-and-inter/) | 获取的 arXiv 作者稿为 7 页，不并入 5–6 页基准 |

“字数”在这里指英文词数，不是字符数。表中的数值是可复现的 **PDF 版面近似词数**：

- 按每页左栏、右栏依次提取，以罗马数字主节标题为边界。
- 修复常见连字符断词和字符连字；主要计入两个及以上英文字母构成的词，另保留 a/A/I。
- 排除标题作者栏、关键词、参考文献、致谢，以及核实过的首页作者/资助脚注。
- 包含小节标题、算法文字、图注及可提取的图表英文标签；纯数字和大部分单字母公式变量不计。
- 浮动图表按实际出现位置归入章节。跨栏图、嵌入图片文字、数学符号和 PDF 字符顺序会造成误差，因此不能把个位数差异解释为有意义的写作差异。

原始 PDF、带页码的提取文本、逐节边界、文件 SHA-256 和计数程序保存在本目录。section_counts.json 对应外部样本，manuscript_counts.json 对应重写后的本稿。

## 2. 各 section 的实际篇幅

| 样本 | 摘要 | 引言 | 系统模型/问题 | 方法 | 仿真/结果 | 结论 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1：对比学习图像传输 | 206 | 377 | 441 | 1,143 | 880 | 75 | 3,122 |
| P2：加权知识库 | 119 | 576 | 516 | 930 | 647 | 78 | 2,866 |
| P3：局部采样语义通信 | 177 | 591 | 701 | 987 | 1,103 | 108 | 3,667 |
| P4：概率语义通信与 RSMA | 190 | 361 | 1,199 | 304 | 452 | 118 | 2,624 |
| P5：NOMA 感知语义通信 | 114 | 938 | 983 | 297 | 105 | 145 | 2,582 |
| 五篇中位数 | 177 | 576 | 701 | 930 | 647 | 108 | 2,866 |
| 本次重写稿 | 164 | 496 | 272 | 617 | 1,048 | 86 | 2,683 |

中位数一行逐列计算，不能将各列中位数相加理解为“中位论文”。本稿数值同样采用 PDF 口径，包含图注和算法文字。

两个边界样本说明不应机械套用均值。P5 花了约 938 词回顾背景，仿真正文只有约 105 词；这不适合本稿需要逐步验证的接收机机制。S1 在两页内合并系统模型与网络设计，并把总结放在结果节末尾，适合短文。S2 的较长作者稿拆成八节，说明五节结构也不是所有 VTC 工作的唯一形式。

## 3. 各节承担的论证任务

### Abstract：一个缺口，一个机制，一个可检查的结果

P1 先指出任务推断与重建质量之间的困难，再给出对比学习和两阶段训练，最后落到数据结果。P4 按“资源冲突 → 优化问题 → 求解办法 → 数值验证”压缩全文。可借鉴的是这一闭合关系，而不是泛泛的 6G 开场或反复宣称优越性。

本稿用约 164 词完成：真实 SINR 不可得且功率随块变化 → 局部能量初始化与迭代最小二乘 → 对 Block-blind 的平均增益及区间 → 两个失效边界。删去摘要中的多组细分结果和未证明的“reconstruction-optimal”。

### Introduction：背景必须最终落到本文要解决的假设

P2 从既有方法依赖的知识库假设转向实际数据偏差，然后引出样本权重反馈。P3 从设备不能一次观察全部数据，推导出局部采样、接收端融合和反馈选位置三个设计需求。引言中的文献不是独立书目清单，而是建立“现有方法为何留下这个问题”的论证。

本稿的段落顺序改为：图像传输中的结构化干扰 → 直接分离和扩散先验两条路线 → ICDM 的全局 SINR 输入及局部失配 → 扩散中间估计为何可用于校准 → 三项贡献及范围。现有 Related Work 并入这里。

### System Model：定义观测、未知量、约束与目标

P4 的模型节较长，是因为它先定义语义压缩、速率拆分、发射功率与计算功率，再形成带约束的优化问题；其后算法自然较短。P1 的模型节则主要把编码、信道、解码及下游任务串起来。两个例子说明：模型节长度取决于问题本身，不应为了接近均值扩写。

本稿的模型只有 AWGN 加块幅度，保留较短篇幅，明确帧归一化、名义功率与实际块能量的区别、接收机知道和不知道什么，并写出高斯观测似然。场景参数移到实验设置，避免混淆一般模型和这次仿真。

### Method：每个设计先有理由，再有公式和实现

P1 的方法依次解释网络组成、核心对比机制和训练目标；P2 先定义加权模块，再解释更新和作用；P4、P5 按优化变量拆成子问题。共同点是公式跟随设计需要出现，最后给出可执行步骤。

本稿扩写为：能量初始化及误差来源 → 固定表的局部插值 → 干净点估计和帧投影 → 非负最小二乘问题及闭式解 → 幅度怎样进入保留的 ICDM 引导式 → 推理流程与计算成本。区分“固定点估计下的最小二乘最优解”和未经证明的联合收敛、图像质量最优性。

### Simulation Results：每组比较回答一个问题

P1 先交代数据、训练和基线，再沿带宽与 SNR 分析结果；P2 先看标签混淆，再比较压缩率与偏差强度；P3 解释不同采样策略及信道条件。可借鉴的是“设置 → 有意义的对照 → 趋势 → 原因或限度”的顺序。

本稿合并原来的 Experimental Setup、Results、Discussion：先给公平比较与聚类统计口径，再依次回答全局 SINR 能否估计、局部功率是否重要、反复校准是否额外有用、何时不应更新或不应调用扩散。最后报告功率异常与实测耗时。两幅图分别展示绝对 PSNR 曲线和带区间的配对增益，均来自现有记录。

### Conclusion：只留下已经得到的答案

P1、P2 的结论约 75–78 词；其作用是回收方法与主要发现。过长的结论容易再次重复引言和结果。

本稿以约 86 词收束：提出了什么接收机、平均提高多少、哪些场景会损失、由此引出的下一步。不把可靠性门控或独立验证写成已经实现的贡献。

## 4. 本稿前后变化与核验

TeXcount 的源码正文口径与上面的 PDF 口径不同，前者不计图中文字，而且会忽略部分自定义宏、公式和算法命令。它适合比较本稿前后，但不与外部 PDF 计数直接混用：

| 部分 | 旧稿源码正文词数 | 新稿源码正文词数 |
|---|---:|---:|
| Introduction | 414 | 496 |
| Related Work | 137 | 已融入引言 |
| System Model | 169 | 232 |
| Method | 283 | 539 |
| Experimental Setup + Results + Discussion | 231 + 556 + 220 = 1,007 | 946 |
| Conclusion | 124 | 87 |

旧稿有 8 个主体章节，新稿为 5 个。增加方法所需细节，压缩重复讨论；由内部编号主导的结果叙述改成 Direct、Global-SINR、Global-blind、Block-blind、SC-ICDM，保留 A0–A4 作为复现对应关系。

全部结果仍来自既有 200 对图像的探索性评估。新增图逐点检查每个“接收机 × 场景 × SINR”单元有 600 条记录；置信区间读取既有图像对聚类 bootstrap 审计文件。没有生成或填补新实验结果。

原 IEEE LaTeX 稿已保留为 paper/ieee_vtc2027/main_before_vtc_rewrite.tex。本次权威正文为 paper/ieee_vtc2027/main.tex。使用标准 IEEEtran conference 类，5 页，未修改类文件、字号或页边距。

VTC2027-Spring 的[官方征稿说明](https://events.vtsociety.org/vtc2027-spring/call-for-papers-2/)要求五页完整论文，并规定额外页数的处理方式；它并未给出每节英文词数配额。以上篇幅选择是基于公开全文和本研究内容作出的编辑判断。

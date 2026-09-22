# 检索、筛选与决策溯源

日期：2026-09-21。执行目的：为 ICDM/RG-ICDM 下一步和 VTC 2027 选题提供依据。没有提交新 GPU 作业，没有运行新性能实验，没有恢复暂停的监控自动任务。

## 检索方法

使用 nature-academic-search 的多来源检索与核验流程，以及 brainstorming-research-ideas 的失败分析和候选筛选。当前工具列表未提供专用学术 MCP，采用网页检索直接访问 arXiv、CVF、PMLR、IEEE、作者项目页/机构记录。二级聚合页只用来发现题目或标识符；最终矩阵均链接原始来源。未执行引用次数和独立他引审计，不据此排序论文质量。

检索范围：2018 年基础 DeepJSCC 工作至检索日可获得的 2026 年工作；中心是图像通信中的接收端恢复、扩散逆问题、未知干扰、计算控制与风险校准。文本专用、纯信道生成、与当前问题不匹配的大系统论文不作为核心证据。不是穷尽式系统综述，未伪造 PRISMA 原始命中量。

实际使用的查询主题与代表查询：

| 查询组 | 查询示例 | 用途 |
|---|---|---|
| 经典基线 | Deep joint source channel coding wireless image transmission; SwinJSCC; CDDM channel denoising diffusion | 确认表示与去噪基线 |
| 干扰主线 | ICDM interference cancellation diffusion; SecDiff diffusion secure JSCC | 定位最接近研究 |
| 生成通信 | GenerativeJSCC; SING semantic image communications; DiffCom channel received signal natural condition | 避免重复已有生成恢复贡献 |
| 盲逆问题 | BlindDPS; GibbsDDRM; Fast Diffusion EM; Tweedie Moment Projected Diffusions | 核验交替估计、矩近似与理论边界 |
| 自适应计算 | semantic communication diffusion adaptive receiver selection early exit risk calibration | 查找选择、早停与预算分配先例 |
| 新近相关 | training-free semantic communication; latent diffusion denoising receiver; FAST; channel-aware preemptive scheduling | 防止把尺度适配和自适应步数当新问题 |
| 风险 | Learn then Test; Conformal Risk Control; non-monotonic losses | 找可用于有限策略族的可靠性工具 |
| 评价 | perception distortion tradeoff; wireless hallucination | 分开像素、感知和语义主张 |
| 会议 | site:events.vtsociety.org/vtc2027-spring papers pages tracks | 核实时间、篇幅和主题 |

结果采用标题与 arXiv/DOI 标识符去重。对 CDDM 会议与期刊扩展版本不重复计数；区分 SING 与检索中无关的同名工作。30 篇纳入矩阵，12 篇关键全文部分阅读，18 篇摘要筛选。另有命中未纳入矩阵，例如仅找到二级记录的 one-step denoising 方法、尚未充分核实的 foundation-model 通信工作，以及出版日期在检索日之后的 early-exit 条目。

访问失败：部分指定版本 HTML 与 CVF PDF 重开返回错误；改用可访问 arXiv HTML、正式会议页面及已返回的全文检索段。Fast Diffusion EM 的 F 标签只代表已读到返回的 §3 式 31–33 与 §4.1 段落，不代表下载并通读其全部 PDF。LTT 作者 PDF 重开失败，改用 arXiv 全文 §1–3。未将无法获取的全文标为已精读。

## 12 个原始候选与筛选结果

| 候选 | 保留程度 | 判断理由 |
|---|---|---|
| 1. 冻结 B_COUPLED 后立即扩大实验 | 否 | 尚未胜过 A4；先补强对照 |
| 2. A4 加同一旁路门控 | 必做控制 | 很可能解释高 SINR 收益；即使胜出也未必单独有足够创新 |
| 3. 盲总扰动功率 CDDM | 必做控制 | 现有 N0-only 对照缺少合理适配；oracle 差距给出直接诊断线索 |
| 4. D/G/S 三类接收模型盲选择 | 首选候选 | 复用资产；但必须证明 y 中存在可预测的收益信号 |
| 5. 独立数据上的相对损失风险校准 | 与 4 条件结合 | 有明确定义与否证标准；不是原创统计理论，样本需求可能成为瓶颈 |
| 6. 未知干扰变化点/随机持续长度 | 第二候选 | 更接近实际未知轮廓；经典分段方法本身不新 |
| 7. 目标身份侧信息与可辨识性 | 长期候选 | 同分布干扰揭示本质限制；需新的预算与任务设计 |
| 8. EM 幅度更新与更复杂双反馈 | 暂停扩展 | 已有近邻方法多，现有反馈优势不足以抵消复杂度 |
| 9. TMPD 式完整后验协方差 | 暂缓 | 数学上有启发；Jacobian/求解成本高且神经近似需验证 |
| 10. 大模型语义裁判或 caption 控制 | 暂缓 | 已有 Adaptive Requests；额外文本与裁判成本会改变问题 |
| 11. RL 或多用户资源调度 | 暂缓 | 引入训练、状态分布和新基线，近期难闭合 |
| 12. 更大 diffusion/全链路重训练或完整 V2X 仿真 | 暂缓 | 容易把机制问题混入表示容量；截止期内成本过高 |

三个入围方向的实验、反对意见和继续条件详见 RESEARCH_DECISION_ZH.md。排序是本项目资产与期限下的研究判断，不是对方向学术价值的客观打分。

## 本地证据与重新计算

- 最新分析：research/rg_icdm/rg_validation_analysis.json。
- 新运行元数据：research/rg_icdm/rg_validation_metadata.json。
- 旧基线：research/published_baselines_validation_v1/means.csv、by_sinr.csv、metadata.json。
- 基线移植边界：research/PUBLISHED_BASELINE_PROTOCOL.md。
- 原始方案：research/rg_icdm/user_design.txt。

重新计算：B_COUPLED−A3 = 0.5600123 dB；10 dB 条件对该差值的贡献约 0.3914257 dB，占约 69.8959%；A4 替换 10 dB 条件为直接解码的假想值约 20.8184605 dB；B_COUPLED 相对 A4 平均 MSE 增加约 4.7873%；相对同轮 A3 接收耗时降低约 7.9465%。后者是同轮描述性测量，不是跨 GPU 排名。

风险样本量示例独立计算：1−(0.05/20)^(1/64)=0.0893682；ceil(log(0.05/20)/log(0.95))=117。仅适用于说明文中相应 Bernoulli/i.i.d./有限策略族设定，不是现有实验认证结果。

旧 STATUS.md 的早期总结建议继续 RG，并含过时监控频率；本轮没有覆盖历史记录。本报告取代“直接冻结 B_COUPLED”的研究建议；自动任务实际配置处于 PAUSED，本轮不改变它。

## 会议核验

- [当前 CFP 网页](https://events.vtsociety.org/vtc2027-spring/call-for-papers-2/)：regular extended deadline 2026-09-30；建议 5 页，支付额外费用最多 7 页。
- [会议首页](https://events.vtsociety.org/vtc2027-spring/)：2027-06-20 至 23，Hamburg。
- [Workshop 页面](https://events.vtsociety.org/vtc2027-spring/workshops/)：当前列出 workshop paper deadline 2026-12-15，具体主题需继续核对。
- 检索中旧 CFP PDF 仍显示 2026-09-01，采用明确标记 EXTENDED 的当前网页日期；未假设论文系统截止时区。

本计划默认 Spring；用户如指 Fall，时间安排需要调整，研究公平性与选题判断不变。没有执行投稿、联系作者或其他外部沟通。

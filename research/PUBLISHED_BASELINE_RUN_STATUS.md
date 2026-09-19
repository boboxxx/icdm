# 已发表基线实验：执行状态

记录时间：2026-09-19。以下为最终状态，不是自动更新的监控页面。

## 已完成

- 阅读并核验 SwinJSCC（TCCN 2025）、CDDM（TWC 2024）、ICDM（JSAC 2026）及 ISEC（AISTATS 2023）的相关方法、实验和指标。
- 把 A1–A4 与外部已发表算法分开；新增 CDDM 同骨干接收算法对照，并明确它不是原文三阶段完整系统的复现。
- 新增 PSNR、图像 MSE、LPIPS-VGG v0.1、四尺度 MS-SSIM 的逐图评价；修正新评价中的末尺度重复乘法，不改动旧实验代码。
- 四项单元测试通过；实际 VGG 指标 identity gate 通过。
- 四对图像的小规模实验完成：180 条完整配对记录，全部有限，PSNR/MSE 换算检查通过。它只证明工程链路可运行，不是论文主结果。
- 正式实验源文件哈希复核通过，并保存 source_snapshot.tar.gz。

## 正式实验已完成

- 主机：sheng，RTX 4090。
- 远端状态为 complete，共 11,520 / 11,520 条记录；总耗时约 48 分 20 秒。
- 64 对验证图像 × 2 固定种子 × 6 SINR × 3 干扰形态 × 5 条接收条件。
- 五条件：SwinJSCC 直接解码、CDDM-RX 仅热噪声、CDDM-RX 全局 SINR oracle、ICDM 全局 SINR oracle、盲 SC-ICDM A4。
- 远端目录：`/mnt/d/ICDM_SC_runs/published_baselines_validation_v1`。
- 远端程序：`/home/sheng/ICDM_SC_20260915/scripts/run_published_baselines.py`。
- 后续检测到另一项目在同一主机训练，未修改或中止其任务。耗时只作共享主机下的诊断记录，不用来宣称效率优势。

完整记录已经拉取并通过审计；已生成总体均值、分 SINR、分干扰形态、逐单元结果和中文分析。`baseline_validation_note.tex` 仍需写入最终结果表后再编译。

## 已执行的收尾步骤

1. metadata.json 状态确认为 complete，记录总数为 11,520。
2. 完整目录已保存到 `research/published_baselines_validation_v1`。
3. 完整性、唯一配对、指标一致性及 smoke 重复误差检查全部通过；已生成 means.csv、strata.csv、by_sinr.csv、by_profile.csv、summary.json 和 RESULTS.md。
4. 按用户要求，结果分析只报告均值、严格配对差值和逐配对更优比例，不计算置信区间，不报告标准差或 mean±std。
5. 已单独记录 CDDM oracle 优于 A4 的条件，并明确 PSNR 与 MSE 是同一像素误差族的两种汇总，不能当作两项独立证据。
6. IEEE 验证说明稿已加入最终结果表并成功编译为 3 页 PDF；逐页检查未发现可见裁切或重叠。更新正式论文时，仍必须将这 64 对验证结果与历史 200 对探索性消融清晰分开。

本轮不执行新的独立测试，不重新训练网络，不取消其他项目的任务。

详细依据见 [基线与指标协议](PUBLISHED_BASELINE_PROTOCOL.md)。

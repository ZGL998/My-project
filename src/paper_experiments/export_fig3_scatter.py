# -*- coding: utf-8 -*-
"""
Figure 3 散点图数据导出：预测 CSCS vs 真实 CSCS（253 名 COPD 患者）
========================================================================
模型：4D-XGB（四维超图表征 + XGBoost 读出）
协议：corrected component-blind（已扣留 severity 维度分量 + 28 个衍生图统计量）
评估：分层 5 折交叉验证，10 个随机种子；每名患者在"处于测试折"时得到一次 OOF 预测
池化：10 个种子的 OOF 预测按患者求平均 → 每名患者 1 个池化预测值 → 253 个点

产出：
  - fig3_scatter_points.csv（253 行，两列 actual_cscs / predicted_cscs）
  - 自检打印：R²(pooled)、Pearson r、MAE、RMSE
  - 线性拟合：slope / intercept

依赖：仅 numpy + pandas
"""
import numpy as np
import pandas as pd

# 10 个随机种子（与论文 P10 口径一致）
SEEDS = [42, 123, 456, 7, 13, 99, 202, 404, 777, 1234]


# ============================================================
# 【替换点 1】加载真实 CSCS —— 请替换成你自己的加载逻辑
# ============================================================
def load_actual_cscs():
    """返回 253 名患者的真实 CSCS，shape (253,) 的 np.ndarray（float）。"""
    # TODO(替换)：改成你自己的真实 CSCS 加载逻辑
    raise NotImplementedError("【替换点 1】请在此填入 load_actual_cscs 的加载逻辑")


# ============================================================
# 【替换点 2】加载某种子下的 OOF 预测 —— 请替换成你自己的加载逻辑
# ============================================================
def load_oof_predictions(seed):
    """返回该种子下 253 名患者的 OOF 预测，shape (253,) 的 np.ndarray（float）。

    注意：
      - OOF = 每名患者仅在其处于"测试折"时被预测一次；
      - 同一患者在不同种子下，顺序必须与 load_actual_cscs() 保持一致（按患者对齐）。
    """
    # TODO(替换)：改成你自己的 OOF 预测加载逻辑
    raise NotImplementedError("【替换点 2】请在此填入 load_oof_predictions 的加载逻辑")


def r2_score(y_true, y_pred):
    """pooled 口径 R²：先池化预测，再对全体 253 点算一次 R²。"""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot


def main():
    # ---- 1) 真实值 ----
    actual = np.asarray(load_actual_cscs(), dtype=float)
    assert actual.shape == (253,), f"真实 CSCS 应为 (253,)，实际 {actual.shape}"

    # ---- 2) 逐种子 OOF 预测，按患者对齐后堆叠成 (10, 253) ----
    preds = np.stack([np.asarray(load_oof_predictions(s), dtype=float) for s in SEEDS])
    assert preds.shape == (len(SEEDS), 253), f"OOF 预测应为 ({len(SEEDS)}, 253)，实际 {preds.shape}"

    # ---- 3) 10 种子池化：按患者求平均 ----
    pred_pooled = preds.mean(axis=0)

    # ---- 4) 自检指标（pooled 口径）----
    r2 = r2_score(actual, pred_pooled)
    r = float(np.corrcoef(actual, pred_pooled)[0, 1])          # Pearson r（只用 numpy）
    mae = float(np.mean(np.abs(actual - pred_pooled)))
    rmse = float(np.sqrt(np.mean((actual - pred_pooled) ** 2)))

    print("=" * 60)
    print("自检指标（pooled 口径，10 种子池化后）")
    print("=" * 60)
    print(f"R² (pooled) = {r2:.4f}   期望 0.043")
    print(f"Pearson r   = {r:.4f}   期望 0.278")
    print(f"MAE         = {mae:.4f}   期望 0.149")
    print(f"RMSE        = {rmse:.4f}   期望 0.184")

    # ---- 诊断：逐种子 R² 再取均值（用于辨别口径）----
    per_seed_r2 = [r2_score(actual, preds[i]) for i in range(len(SEEDS))]
    print("-" * 60)
    print(f"[诊断] 逐种子 R² 均值 = {float(np.mean(per_seed_r2)):.4f}"
          f"（若口径错成逐种子均值，此值会接近 -0.013 而非 pooled 的 0.043）")

    # ---- 5) 线性拟合 slope / intercept ----
    slope, intercept = np.polyfit(actual, pred_pooled, 1)
    print("-" * 60)
    print(f"线性拟合: slope = {slope:.4f}, intercept = {intercept:.4f}")

    # ---- 6) 输出 CSV ----
    df = pd.DataFrame({"actual_cscs": actual, "predicted_cscs": pred_pooled})
    df.to_csv("fig3_scatter_points.csv", index=False)
    print("-" * 60)
    print(f"已写出 fig3_scatter_points.csv（{len(df)} 行，两列 actual_cscs / predicted_cscs）")


if __name__ == "__main__":
    main()

# 问题2稳定版与SOC自适应验证说明

## 稳定复现基准

- 评估日期：2025-02-01 至 2025-12-31，共334天。
- 每天仅在0:00制定一次全天计划，日内不修改购电计划。
- 固定80%历史残差分位数基准总费用：14,185,416.777032元。
- 计划购电费：13,461,747.304540元。
- 紧急购电量：119,533.118918 kWh。
- 紧急购电费：723,669.472492元。
- LP与显式禁止同时充放电的MILP结果一致，内置基准核验通过。

## SOC自适应验证

SOC自适应分位数规则为：

```text
q_d = clip(q0 + alpha * (6000 - SOC_d,0) / 4800, 0.50, 0.95)
```

验证采用严格时间切分：2025年2月至6月用于选择参数，2025年7月至12月作为不参与调参的样本外验证期。粗网格和局部精搜索均保留在仓库的验证程序及JSON结果中。

训练期精搜索选择 `(q0, alpha) = (0.92, 0.14)`。相对固定80%基准，该规则在样本外验证期降低总费用25,151.68元；2月至12月汇总总费用为14,132,501.60元，比稳定基准低52,915.17元。该结果用于验证SOC信息的增量价值，稳定复现基准仍保留为14,185,416.78元版本。

## 复现文件

- `outputs/problem2_attached_baseline.json`
- `outputs/problem2_attached_baseline.xlsx`
- `validate_soc_quantile.py`
- `outputs/soc_quantile_validation/soc_quantile_validation.json`
- `outputs/soc_quantile_fine/soc_quantile_validation.json`


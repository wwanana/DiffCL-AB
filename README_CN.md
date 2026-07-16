# ProtAttBA-MambaCL 中文说明

本项目是在 ProtAttBA 基础上完成的序列级抗体–抗原结合亲和力变化预测模型。输入包括野生型抗体、野生型抗原、突变型抗体和突变型抗原四条序列，输出突变导致的结合自由能变化 ΔΔG。

## 主要改进

- 冻结 ESM2-650M 主干，提取残基级表示；
- 保留抗体–抗原交互注意力主分支；
- 增加 Mamba 并行分支，学习长程序列依赖；
- 使用门控交互融合主分支与 Mamba 分支；
- 引入两类对比学习约束；
- 通过差异感知与状态感知特征融合预测突变前后结合变化；
- 支持断点续跑、仅保存权重、早停和五折交叉验证；
- 输出 MSE、RMSE、MAE、Pearson、Spearman 和 R²。

## 环境安装

```bash
conda env create -f environment.yml
conda activate protab_mamba

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -U packaging ninja
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

## 下载 ESM2

```bash
python scripts/download_esm2.py --output_dir ./model/esm2_650m
```

不要把 ESM2 权重、训练 checkpoint 和结果日志直接提交到 GitHub。

## 三个数据集五折交叉验证

```bash
bash scripts/run_cv5_ab1101.sh
bash scripts/run_cv5_ab645.sh
bash scripts/run_cv5_s1131.sh
```

默认主要参数：随机种子 42、最大 120 epochs、早停 patience 20、学习率 2e-5、weight decay 1e-4、batch size 2、梯度累积 4、冻结 ESM2、启用 Mamba 和双对比学习。

## 发布前必须检查

- 删除 checkpoint、结果目录、日志、缓存和模型权重；
- 检查代码中是否包含服务器绝对路径、账号、密钥或令牌；
- 将所有 `argparse type=bool` 改为可靠的 `str2bool`；
- 用实际训练环境生成 `requirements-lock.txt`；
- 确认原始 ProtAttBA 代码许可证、数据许可证和 ESM2 权重许可证；
- 更新 `CITATION.cff` 中的作者、论文标题、仓库地址和 DOI。

详细内容请阅读英文版 [`README.md`](README.md)。

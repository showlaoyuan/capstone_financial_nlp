# Capstone Financial NLP

基于金融领域自然语言处理的毕业设计项目，主要包括金融文本数据集整理、数据预处理、FinBERT 模型训练与评估，以及金融情感分析任务实现。

## Project Overview

本项目围绕金融文本分析展开，目标是构建一个完整的金融自然语言处理实验流程，包括：

- 金融数据集下载与整理
- 数据检查与格式统一
- 探索性数据分析（EDA）
- FinBERT 模型训练
- 模型评估与结果分析

## Project Structure

```bash
capstone_financial_nlp/
├─ app/                         # 项目主要代码
├─ data/                        # 数据集与处理结果
│  ├─ metadata/                 # 数据说明或元数据
│  ├─ processed/                # 处理后数据
│  ├─ raw/                      # 原始数据
│  ├─ unified/                  # 统一整理后的数据
│  ├─ dataset_samples.txt       # 数据样本说明
│  ├─ dataset_summary.csv       # 数据集汇总信息
│  ├─ financial_sentiment.csv   # 金融情感数据
│  └─ fiqa_raw.csv              # FiQA 原始数据
├─ results/                     # 实验结果
│  ├─ figures/                  # 图表
│  └─ reports/                  # 分析报告
├─ README.md                    # 项目说明
├─ .gitignore                   # Git 忽略规则
└─ LICENSE                      # 开源许可证
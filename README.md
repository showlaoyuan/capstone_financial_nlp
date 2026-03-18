## Project Structure

```bash
capstone_financial_nlp/
├─ app/                          # 项目代码
│  └─ api/                       # 数据处理、训练、评估等脚本
├─ data/                         # 数据集与处理结果
│  ├─ metadata/                  # 数据说明与元数据
│  │  ├─ dataset_samples.txt
│  │  └─ dataset_summary.csv
│  ├─ processed/                 # 处理后数据
│  │  └─ financial_sentiment.csv
│  ├─ raw/                       # 原始数据
│  │  ├─ finqa/
│  │  ├─ fiqa/
│  │  │  ├─ fiqa_raw.csv
│  │  │  ├─ fiqa_train.csv
│  │  │  ├─ fiqa_test.csv
│  │  │  └─ fiqa_validation.csv
│  │  └─ fpb/
│  └─ unified/                   # 统一整理后的数据
│     ├─ finqa_test_200.csv
│     ├─ finqa_train.csv
│     ├─ fiqa_test_200.csv
│     ├─ fiqa_train.csv
│     ├─ fpb_test_200.csv
│     └─ fpb_train.csv
├─ results/                      # 实验结果
│  ├─ figures/                   # 图表结果
│  ├─ models/                    # 训练后的模型
│  └─ reports/                   # 分析报告
├─ README.md                     # 项目说明
├─ .gitignore                    # Git 忽略规则
└─ LICENSE                       # 开源许可证
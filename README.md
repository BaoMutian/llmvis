# LLM Inference Visualizer

一个用于可视化大语言模型推理过程的 Web 工具，帮助理解 LLM 内部机制。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.40+-orange.svg)

## ✨ 功能特性

### 基础功能
- **双模式推理**：支持 Chat（对话）和 Completion（续写）两种模式
- **参数调节**：实时调整 Temperature、Top-K、Top-P、Max Tokens
- **Token 级分析**：点击任意生成的 token 查看详细信息

### 可视化分析

| 分类 | 功能 | 说明 |
|------|------|------|
| **注意力分析** | 热力图 | 查看指定层的注意力权重分布 |
| | 多头对比 | 对比不同注意力头的模式 |
| | 注意力头熵 | 识别聚焦型和分散型注意力头 |
| **隐藏状态** | 层间相似度 | 观察信息在层间的流动 |
| | 残差流 | 理解每层的贡献 |
| | Logits Lens | 观察模型何时确定答案 |
| **Token 分析** | 概率分布 | Top-N 候选 token 的概率 |
| | Embedding 投影 | 2D/3D 语义空间可视化 |
| | 激活分布 | MLP 和 Attention 层激活统计 |
| **归因分析** | 熵曲线 | 模型不确定性分布（支持排序） |
| | 置信度 | 每个 token 的确信程度 |
| | 输入归因 | 输入对输出的影响程度 |

### Token 颜色模式
- **默认**：高亮非最高概率的采样 token（黄色）
- **熵值**：按熵值着色（高熵红色，低熵绿色）
- **概率**：按生成概率着色
- **Logit**：按 logit 差距着色

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA 兼容的 GPU（推荐 16GB+ 显存）
- 本地 LLM 模型（如 Qwen、LLaMA 等）

### 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd llmvis

# 创建虚拟环境（推荐）
conda create -n llmvis python=3.10
conda activate llmvis

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
python app.py
```

启动后访问 http://localhost:5050

### 加载模型

1. 在页面左上角"模型"区域输入本地模型路径，例如：
   ```
   ~/models/qwen3-4b-instruct
   ```
2. 点击"加载"按钮
3. 等待状态变为"模型已加载"

## 📖 使用指南

### 基本流程

1. **加载模型**：输入模型路径并加载
2. **选择模式**：Chat（问答）或 Completion（续写）
3. **输入内容**：在输入框输入问题或文本
4. **调整参数**：根据需要调整采样参数
5. **生成**：点击"生成"按钮
6. **分析**：
   - 点击任意生成的 token 查看概率分布
   - 切换右侧标签页查看不同维度的分析
   - 调整注意力层/头查看不同层的注意力模式

### 参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| Temperature | 0-2 | 控制随机性，0=确定性，越高越随机 |
| Top-K | 1-100 | 只从概率最高的 K 个 token 中采样 |
| Top-P | 0-1 | 只从累积概率达到 P 的 token 中采样 |
| Max Tokens | 16-512 | 最大生成 token 数 |

### 快捷操作

- **Enter**：快速发送（在输入框中）
- **拖动分隔条**：调整右侧面板宽度
- **2D/3D 切换**：在 Embedding 投影页面切换维度

## 🏗️ 项目结构

```
llmvis/
├── app.py              # Flask 主应用
├── model_service.py    # 模型加载与推理服务
├── analysis.py         # 统计分析函数
├── requirements.txt    # Python 依赖
├── templates/
│   └── index.html      # 前端页面
└── static/
    ├── css/
    │   └── style.css   # 样式
    └── js/
        ├── api.js      # API 调用
        ├── charts.js   # ECharts 图表
        └── main.js     # 主逻辑
```

## 🔧 配置

### 修改端口

编辑 `app.py` 最后一行：

```python
app.run(host="0.0.0.0", port=5050, debug=True)
```

### 支持的模型

理论上支持所有 Hugging Face Transformers 格式的因果语言模型：
- Qwen 系列
- LLaMA 系列
- Mistral 系列
- 其他兼容模型

## ⚠️ 注意事项

1. **显存占用**：逐 token 生成并保存注意力会占用较多显存
2. **生成速度**：由于需要获取每一步的注意力，速度比正常推理慢
3. **模型兼容性**：部分模型的 embedding 层命名可能不同，如遇报错请提 issue

## 📝 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！


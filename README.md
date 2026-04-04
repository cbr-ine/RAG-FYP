# 🧠 Reasoning RAG

基于多跳检索与推理的问答系统，支持复杂问题分解、多跳证据检索与 LLM 答案生成。

---

## 📌 功能特性

- **自动复杂度分析**：判断问题是否需要分解
- **查询分解**：将复杂问题拆解为多个子查询（由 DeepSeek LLM 驱动，含规则回退）
- **多跳检索**：基于 FAISS 向量库进行多轮语义检索
- **证据整合与验证**：对检索结果打分、去重、交叉验证
- **答案生成**：由 DeepSeek LLM 合成最终答案（含简单回退方案）
- **Web 界面**：基于 Flask 的聊天界面，可在浏览器中使用
- **命令行交互**：支持终端直接问答

---

## 🗂️ 项目结构

```
reasoning_rag/
├── app.py                 # Flask Web 服务
├── main.py                # 命令行入口（构建索引 / 评估 / 演示 / 交互）
├── interactive.py         # 独立交互式问答脚本
├── reasoning_rag.py       # 主系统集成
├── config.py              # 全局配置
├── data_loader.py         # BioASQ 数据集加载
├── embedder.py            # 文本嵌入（sentence-transformers）
├── vector_store.py        # FAISS 向量存储
├── multi_hop_retriever.py # 多跳检索
├── query_analyzer.py      # 问题复杂度分析
├── query_decomposer.py    # 查询分解（DeepSeek LLM）
├── evidence_integrator.py # 证据整合与验证
├── answer_generator.py    # 答案生成（DeepSeek LLM）
├── evaluator.py           # 评估指标计算
├── requirements.txt
└── templates/
    └── index.html         # Web 前端页面
```

---

## ⚙️ 环境要求

- Python 3.9+
- （可选）[DeepSeek API Key](https://platform.deepseek.com/) — 不填则退回规则/简单合成模式

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/reasoning_rag.git
cd reasoning_rag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 DeepSeek API Key（可选）

不配置也能运行，但查询分解和答案生成将使用规则/简单模式，效果会下降。

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="your_api_key_here"

# Windows CMD
set DEEPSEEK_API_KEY=your_api_key_here

# Windows PowerShell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

### 4. 构建向量索引

**首次使用或数据更新后必须执行此步骤。**

```bash
# 推荐：使用完整数据集构建索引（用于 Web / 交互模式）
python main.py --mode build --full-index

# 仅用训练集构建（用于评估，避免数据泄露）
python main.py --mode build
```

索引默认保存至 `./bioasq_index.pkl`，可通过 `--index-path` 指定路径：

```bash
python main.py --mode build --full-index --index-path ./my_index.pkl
```

---

## 💻 使用方式

### 方式一：Web 界面（推荐）

```bash
python app.py
```

启动后在浏览器打开 [http://localhost:5000](http://localhost:5000)

> 如果索引文件不在默认路径，可通过环境变量指定：
> ```bash
> # Linux / macOS
> export RAG_INDEX_PATH=/path/to/your_index.pkl
>
> # Windows CMD
> set RAG_INDEX_PATH=C:\path\to\your_index.pkl
> ```

---

### 方式二：命令行交互

```bash
python main.py --mode interactive
```

或使用独立脚本：

```bash
python interactive.py
```

---

### 方式三：演示模式（随机抽取测试集问题）

```bash
python main.py --mode demo --demo-size 3
```

---

### 方式四：评估模式

```bash
python main.py --mode eval --eval-size 50
```

> ⚠️ 评估模式请勿使用 `--full-index` 构建的索引，否则测试集答案已在索引中，评分会虚高。

---

## 🔧 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `demo` | 运行模式：`build` / `eval` / `demo` / `interactive` / `all` |
| `--index-path` | `./bioasq_index.pkl` | 索引文件保存/加载路径 |
| `--full-index` | 关闭 | 使用完整数据集（train + test）构建索引 |
| `--rebuild-index` | 关闭 | 强制重建索引（即使文件已存在） |
| `--max-passages` | `2000` | 仅训练集模式下的最大段落数 |
| `--eval-size` | `50` | 评估时的问题采样数 |
| `--demo-size` | `3` | 演示模式的问题数量 |

---

## 📊 系统配置

核心参数在 `config.py` 中修改：

```python
EMBEDDING_MODEL         = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL         = 5      # 每次检索返回的段落数
SIMILARITY_THRESHOLD    = 0.5    # 过滤低相关度证据的阈值
MAX_SUBQUERIES          = 4      # 最大子查询数量
MAX_HOPS                = 3      # 最大检索跳数
MAX_EVIDENCE_LENGTH     = 2000   # 证据总长度上限（字符）
MIN_EVIDENCE_SIMILARITY = 0.55   # 低于此值拒绝生成答案
```

---

## 🔄 完整运行流程

```
构建索引                    运行服务/交互
    ↓                            ↓
python main.py           app.py / interactive.py
--mode build                     ↓
    ↓                      用户输入问题
bioasq_index.pkl                 ↓
                          1. 复杂度分析
                          2. 查询分解（LLM）
                          3. 多跳 FAISS 检索
                          4. 证据整合与验证
                          5. 答案生成（LLM）
                                 ↓
                              返回答案
```

---

## ❓ 常见问题

**Q：启动 `app.py` 后提示 `Index is empty`**

需要先构建索引：
```bash
python main.py --mode build --full-index
```

**Q：没有 DeepSeek API Key 能用吗？**

可以。查询分解会回退到基于规则的分解，答案生成会使用关键句提取模式，功能完整但质量略低。

**Q：想换用其他数据集怎么办？**

修改 `data_loader.py` 中的 `load_bioasq_dataset()`，使 `get_passages()` 返回包含 `text` 字段的字典列表即可，其余模块无需改动。

**Q：如何更换嵌入模型？**

在 `config.py` 中修改 `EMBEDDING_MODEL`，然后重新构建索引（旧索引维度不兼容）：
```bash
python main.py --mode build --full-index --rebuild-index
```

---

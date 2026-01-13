# Open Deep Research (Enhanced Version)

这是一个基于 LangGraph 的深度研究 Agent 项目，在 Open Deep Research 后端的基础上进行了增强，集成了 **Acemap 学术搜索** 和 **知识图谱 (Knowledge Graph)** 技术，以提供更深入、更专业的学术研究报告。

## ✨ 主要特性

1.  **多 Agent 协作架构**: 基于 LangGraph 构建，包含 Supervisor（监督者）、Researcher（研究员）等角色，实现从规划、执行到报告生成的全流程自动化。
2.  **Acemap 学术搜索集成**:
    *   接入 Acemap API，支持对学术论文、作者、引用关系的深度检索。
    *   在生成的报告中自动包含相关的学术文献引用。
3.  **知识图谱增强 (KG-Enhanced Search)**:
    *   利用 **ConceptNet** 知识图谱对搜索查询进行语义扩展。
    *   采用 TF-IDF 风格的算法计算概念相关性，挖掘潜在的语义关联。
    *   支持基于 GAKG (Geoscience Academic Knowledge Graph) 的领域特定扩展（如地学领域）。
4.  **深度迭代研究**: 支持 "Think -> Act -> Reflect" 的循环机制，Agent 会根据中间结果动态调整研究方向。

## 📂 项目结构

```text
DM_deepresearch/
├── server.py                 # FastAPI 后端服务，提供 API 接口
├── src/
│   └── open_deep_research/
│       ├── deep_researcher.py # LangGraph 核心工作流定义 (Supervisor/Researchers)
│       ├── kg_enhanced_search.py # 知识图谱增强搜索核心逻辑
│       └── utils.py           # 工具函数 (Tavily, Acemap Tools 等)
├── ui/
│   └── acemap.html           # 前端可视化界面
├── runs/                     # 运行时输出目录 (报告、JSON结果)
└── ...
```

## 🚀 快速开始

### 1. 环境准备

确保已安装 `uv` 包管理器和 Python 环境。

```bash
# 安装依赖
uv sync
```

### 2. 配置环境变量

复制 `.env.example` (如果存在) 或直接创建 `.env` 文件，并填入必要的 API Keys：

```env
OPENAI_API_KEY=sk-...
ACEMAP_API_KEY=       # 这里可以不填
# 其他可能需要的 Key 
```

### 3. 运行指南

本项目包含三个主要组件，建议在不同的终端窗口中分别运行以下命令：

#### 步骤 1: 启动 LangGraph 开发服务 (可选/调试用)
这会启动 LangGraph Studio 的后端，用于可视化调试 Agent 工作流。

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . langgraph dev --allow-blocking
```

#### 步骤 2: 启动 API 后端服务
这是核心的业务服务器，提供 REST API (`/api/research`) 来触发研究任务。

```bash
uvicorn server:app --reload --port 8001
```

#### 步骤 3: 启动前端页面服务
启动一个简单的 HTTP 服务器来托管 UI 界面。

```bash
python -m http.server 8000
```

### 4. 使用方法

1.  完成上述启动步骤。
2.  打开浏览器访问: `http://localhost:8000/ui/acemap.html`。
3.  在界面输入想要询问的语句（例如："Is nuclear related to climate change?"）。
4.  系统将开始运行：
    *   **后端 (Port 8001)**: 接收请求，触发 Deep Research Agent。
    *   **Agent**: 进行查询扩展 (KG) -> 搜索 (Acemap) -> 总结 -> 生成报告。
5.  运行结果的保存（如最终报告 `final_report.md` 和结果数据 `acemap_results.json`）将保存在 `runs/` 目录下。


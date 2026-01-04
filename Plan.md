# 总体流程

***

### 项目名称：Acemap Deep Research Assistant (ADRA)

### 核心逻辑
不训练模型，而是用 **LLM 作为大脑（Controller）**，用 **Acemap API 作为手眼（Tool）**，用 **GAKG（知识图谱）作为长期记忆/知识库**。

***

### Phase 0: 准备工作 (Infrastructure Setup)

在开始写代码之前，你需要准备好这三样东西：

1.  **Acemap 接口封装 (Tooling)**
    *   **操作**：编写一个 Python 函数 `search_acemap(query, limit=10)`。
    *   **功能**：输入关键词，返回论文列表（包含 Title, Abstract, Authors, Year, Citation Count）。
    *   *Tip*：如果 Acemap API 返回的数据很杂，写一个简单的清洗函数，只保留核心字段，节省 Token。

2.  **GAKG 数据索引 (Knowledge Access)**
    *   **操作**：不需要把整个图谱加载进内存。将 GAKG 的数据存入一个轻量级数据库（如 SQLite 或 Neo4j），或者简单的 Pandas DataFrame（如果数据量允许）。
    *   **工具函数**：编写 `get_entity_neighbors(entity_name)`，输入实体名，返回它的邻居节点（如：输入 "Deep Learning"，返回相关联的 Author, Conference, Concept）。

3.  **LLM 接入 (The Brain)**
    *   **操作**：申请 DeepSeek、Qwen 或 OpenAI 的 API Key。
    *   **框架**：安装 `langgraph` 和 `langchain`。这是实现 Deep Research 状态机的基础。

***

### Phase 1: 搜索前处理 (Pre-processing / Scoping)

这一步是 Deep Research 的精髓：**不要直接拿用户的 Query 去搜，先规划**。

#### 1. 意图识别与定界 (Scope)
*   **输入**：用户的原始 Query（例如：“最近大模型在气象预测中的应用”）。
*   **动作 (LLM Call)**：
    *   设计 Prompt：
        > "你是一个科研助手。用户想研究 [Query]。请分析这个请求，提取出 3-5 个核心搜索子方向（Sub-queries）。如果是模糊概念，请结合知识图谱中的实体类型（如方法、数据集、应用场景）进行拆解。"
    *   *RAG 增强（可选）*：如果 Query 包含专业术语（如 "GCN"），先查 GAKG，把 "GCN" 的定义和一跳邻居作为 Context 给 LLM，帮助它理解这是个算法。
*   **输出 (Plan)**：
    *   生成一份 JSON 格式的 **Research Brief（研究简报）**。
    *   *Example JSON*:
        ```json
        {
          "goal": "调研大模型在气象预测的应用",
          "sub_tasks": [
            "搜索基于 Transformer 的气象预测模型 (如 Pangu-Weather, GraphCast)",
            "搜索气象预测常用的数据集 (如 ERA5)",
            "对比传统数值天气预报(NWP)与AI方法的差异"
          ]
        }
        ```

***

### Phase 2: 执行搜索 (Research / Execution)

这一步是多智能体/多线程并行的核心。你需要用代码循环处理 Phase 1 生成的 `sub_tasks`。

#### 2. 分布式搜索与图谱增强
*   **输入**：Research Brief 中的每一个 `sub_task`。
*   **动作 (Iterative Search)**：
    *   **Step A: 关键词搜索**
        *   LLM 将自然语言 `sub_task` 转化为 Acemap 的查询语句（Boolean Query）。
        *   调用 `search_acemap` 工具。
    *   **Step B: 知识图谱拓展 (Graph Expansion)**
        *   拿到 Step A 的结果后，提取前 3 篇高引论文的 **作者** 和 **关键词**。
        *   调用 `get_entity_neighbors` 工具，在 GAKG 中查找这些作者的**合作者**或这些关键词的**上位词/下位词**。
        *   *为什么要做这个？* 这能发现单纯靠关键词搜不到的潜在相关论文（链路预测的平替）。
    *   **Step C: 社区发现 (Simple Community Detection)**
        *   利用 `networkx` 库，在内存中构建一个小型的“论文-引用”图。
        *   运行简单的 Louvain 算法，将搜索到的 50-100 篇论文分成 3-4 个簇（Cluster）。
        *   *作用*：自动识别流派。

#### 3. 信息清洗与反思 (Reflection)
*   **动作 (LLM Call)**：
    *   把搜到的 Raw Data（摘要列表）喂给 LLM。
    *   Prompt：
        > "请检查以下搜索结果是否满足子任务 [Sub-task] 的要求？如果不满足，请给出新的搜索关键词；如果满足，请总结核心信息。"
    *   **过滤**：去掉不相关的论文，保留 Top-K 高质量结果。

***

### Phase 3: 搜索后处理 (Post-processing / Synthesis)

这一步是将碎片信息转化为结构化知识。

#### 4. RAG 增强生成 (Report Writing)
*   **输入**：所有子任务清洗后的论文摘要、图谱中的实体关系、以及 Phase 2 识别出的社区（Cluster）信息。
*   **动作 (Final LLM Call)**：
    *   Prompt (Long Context):
        > "基于以上所有检索到的信息，撰写一份关于 [原始 Query] 的深度调研报告。
        > 要求：
        > 1. 必须引用具体的论文（标注 [Author, Year]）。
        > 2. 按技术流派（Community）组织结构。
        > 3. 明确指出该领域的发展趋势。"
*   **输出**：Markdown 格式的完整报告。

#### 5. 可视化输出 (UI Display)
*   **动作**：
    *   **力导向图 (Force Graph)**：利用 D3.js 或 ECharts，画出本次搜索涉及的节点（论文、作者、概念）。用不同颜色标记 Phase 2 中发现的社区。
    *   **引用路径**：如果 LLM 提到“A 启发了 B”，在图上高亮 A -> B 的边。

***

### 你的代码实现清单 (To-Do List)

你可以直接照着这个顺序写 Python 代码：

1.  **`tools.py`**:
    *   `def search_acemap(query): ...` (调用 API)
    *   `def query_gakg(entity): ...` (查库/查表)
2.  **`agent.py`** (使用 LangGraph):
    *   定义 `State` (存储 search_results, research_brief)。
    *   定义 `Planner Node` (实现 Phase 1)。
    *   定义 `Worker Node` (实现 Phase 2，可以是一个循环)。
    *   定义 `Writer Node` (实现 Phase 3)。
3.  **`graph_utils.py`**:
    *   `def build_citation_graph(papers): ...`
    *   `def detect_communities(graph): ...` (调用 networkx)
4.  **`main.py`**:
    *   接收用户输入，运行 Graph，打印 Markdown 报告。

### 预期效果 (Expected Outcome)

1.  **广度提升**：用户搜一个词，系统能通过 GAKG 自动联想到相关联的冷门领域，搜索结果不再局限于包含关键词的论文。
2.  **结构化展示**：不再是扔给用户 20 篇论文列表，而是一份带“流派分类”的综述。
3.  **精准度/信噪比**：通过 LLM 的自我反思（Reflection）机制，自动剔除了标题相关但内容无关的噪音数据。
4.  **图挖掘任务完成度**：虽然没训练模型，但你在 Pipeline 中实实在在地使用了 **社区检测 (Community Detection)** 和 **图遍历 (Graph Traversal)** 算法，完美符合“在图结构数据上进行挖掘”的要求。

这个方案**完全可落地**，且逻辑闭环。你可以先用一个简单的 Query 跑通全流程，再慢慢优化 Prompt。

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/156382011/882588b4-ffec-40ad-bda8-fa2f1a340825/AI-3602-Pilot-Project-Qi-Mo-Xiang-Mu-Xian-Dao-Zuo-Ye-CodiMD.pdf)


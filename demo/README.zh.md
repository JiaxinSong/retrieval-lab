# 演示说明

这里存的是五次运行的终端输出，不需要 API key 也能看到整条流水线的效果。每个文件都是用 `| tee demo/<文件名>.txt` 保存的；重新运行需要 `OPENAI_API_KEY` 和 `COHERE_API_KEY`，费用是几美分。英文版见 `README.md`。

| 文件 | 演示什么 |
| --- | --- |
| `04_hybrid_search.txt` | 六个查询上，检索的每个阶段并排对比 |
| `ask_work_mem.txt` | 一个前提错误的问题，回答方式是纠正前提 |
| `ask_multihop.txt` | 一个两部分的问题，两半分别来自不同的小节，各自带引用 |
| `ask_unanswerable.txt` | 一个文档回答不了的问题：分数低，再搜一次，然后拒答 |
| `ask_agentic.txt` | 强制打开 agentic 循环：模型改写查询，用第二次搜索的结果回答 |

---

## 1. `04_hybrid_search.txt`：检索的每个阶段

```bash
python scripts/04_hybrid_search.py
```

**展示什么。** 对每个查询，依次列出每路检索单独的前 5 名、RRF 融合后、加小节上限后、rerank 后的结果。相当于把 `scripts/notes/04-retrieval.md` 里讲的整条流水线摊开来看。

**每一行怎么读。**

```
    4 D v#14 l#4                    18.4.3. Resource Limits
    1   v#4 l#5 i#7        0.6241   19.4.1. Memory
```

- 第一个数字：在这个列表里的位置
- `D`：这个 chunk 来自干扰章节（从不出题）
- `v#14 l#4 i#7`：这个 chunk 在向量、词法、标识符三个列表里各排第几；没有某个标记，说明那一路没返回它
- 最后一块里的小数：Cohere 的 rerank 分数
- 每个查询开头：`lexemes kept / dropped` 是文档频率过滤之后的词法查询；`identifiers` 是标识符检索要找的参数名

**逐个查询看什么。**

| 查询 | 演示了什么 |
| --- | --- |
| `work_mem` | 下划线切分：词法查询只剩 `mem`（`work` 太常见被丢掉了），所以只有标识符检索在找完整的 `work_mem` |
| `difference between work_mem and maintenance_work_mem` | 两个参数共享词素 `mem`，单靠词法检索几乎分不开它们 |
| `Is work_mem a per-connection limit?` | 干扰项（`18.4.1`、`18.4.3`）进了向量和词法的前 5；rerank 调整了 5 个位置中的 4 个 |
| `why does the database slow down after many updates` | 向量和词法找到的 chunk 完全不同（并集 40 个，交集 0 个），也没有标识符。Rerank 最高分只有 **0.40**，排第一的还是个干扰项：这是召回失败，而低分就是信号 |
| `what does autovacuum_vacuum_scale_factor control` | 检索本来就处理得很好的查询：rerank 只调整了 1 个位置 |
| `preventing transaction id wraparound` | 小节霸榜：向量检索的前 5 全部来自 `24.1.5` 这一个小节。小节上限给 `19.10. Automatic Vacuuming` 腾出了位置；rerank 最高分 **0.90** |

覆盖得好的问题 0.90，没有好答案的问题 0.40，这两个分数就是"用 rerank 分数触发 agentic 重试"的依据。

---

## 2. `ask_work_mem.txt`：纠正错误的前提

```bash
python scripts/08_agent.py --ask "Is work_mem a per-connection limit?"
```

**展示什么。** 这个问题的前提是错的。答案以 "No" 开头，再解释真正的规则：这个限制按每个排序或哈希操作计算，多个操作、多个会话、多个并行 worker 都可以各自使用这么多内存。

**看什么。**

- `top score 0.62` 高于 0.5 的门槛，所以 `path: direct`：只调用一次模型，不给搜索工具
- 每个说法都带 `[n]` 引用，`Sources` 把每个编号对应到小节和 URL
- 这正是第 06 步校准 judge 时要求的行为：一段文字可以通过纠正前提来回答问题，而不必复述问题原话（`scripts/notes/zh/06-golden-set.zh.md`，*第一次校准失败了*）

---

## 3. `ask_multihop.txt`：两半答案，两个小节

```bash
python scripts/08_agent.py --ask "How much RAM should I give shared_buffers, and what does effective_cache_size actually do?"
```

**展示什么。** 标准答案集里的一道并列多跳题（hw9）。没有任何一个 chunk 能同时回答两部分：`shared_buffers` 该设多大在 `19.4.1. Memory` 里，`effective_cache_size` 的作用在 `19.7.2. Planner Cost Constants` 里。

**看什么。**

- `[1]` 支撑前一半，`[2]` 支撑后一半，两个来源是不同的小节
- 第一次检索就已经把两半都找到了（`top score 0.59`，`path: direct`），因为两个参数名都在问题里。这也是为什么并列多跳题从来用不到 agentic 循环（`scripts/notes/zh/08-agent.zh.md`）

---

## 4. `ask_unanswerable.txt`：拒答，而不是编造

```bash
python scripts/08_agent.py --ask "How do I shard a table across nodes with Citus?"
```

**展示什么。** Citus 是 PostgreSQL 的一个扩展，但 PostgreSQL 手册里没有它的文档。检索返回了一些看起来相关的、讲分区的 chunk，这种"差一点就对"的情况最容易诱使模型自信地编出一个答案。

**看什么。**

- `top score 0.12`，低于 0.5，所以模型拿到了搜索工具：`path: agentic`
- 它改写查询又搜了一次，分数也只到 0.20
- 最后回答 `answerable: False`，说明文档没有涉及这个问题

**这个演示暴露的一个已知问题。** 拒答时它引用了读过的全部九段（`[1]` 到 `[9]`），而其实没有一段支持任何答案。拒答本应不带引用。修复方法是在 answer 工具里加一条规则："`answerable` 为 false 时不能有引用"。目前还没改，因为改了就要重跑第 08 步。

---

## 5. `ask_agentic.txt`：agentic 循环

```bash
python scripts/08_agent.py --ask "EXPLAIN ANALYZE shows row estimates far from the actual rows -- what setting helps the planner estimate better?" --threshold 1.0
```

**展示什么。** `--threshold 1.0` 让模型在每道题上都拿到搜索工具，不管分数多少。这道题是 hw5，也就是第 08 步强制循环时唯一救回来的那道。

**看什么。**

- 第一次搜索得分 0.76；在默认门槛下，这道题会走直接路径
- 模型选择再搜一次，改写后的查询加上了文档里的术语：`default_statistics_target`、`SET STATISTICS`
- 答案引用的是 `[6][7][9]`，编号超出了第一次搜索的 5 个，所以来自第二次搜索。段落编号在多次搜索之间连续，每个编号依然只对应一段
- 怎么解读：在第 08 步标准答案集的运行里，检查过第一次搜索返回的段落，没有一段提到 `default_statistics_target`，所以这个词是模型凭自己的知识带进来的。这是查询扩展，而不是依赖第一次搜索证据的第二跳（`scripts/notes/zh/08-agent.zh.md`，*救回的那道是顺序题*）

---

## 面试时哪个问题配哪个演示

| 问题 | 演示 |
| --- | --- |
| "为什么要三路检索？" / "为什么要标识符检索？" | 1，`work_mem` 那个查询 |
| "Rerank 带来了什么？" | 1，wraparound 和 slow down 两个查询（0.90 对 0.40） |
| "你怎么处理幻觉？" | 4 |
| "能回答多部分的问题吗？" | 3 |
| "Agent 到底做了什么？" | 5 |
| "演示一下它能用" | 先 2，再 3 |

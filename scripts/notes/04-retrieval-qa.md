# 04 — Retrieval Q&A

自测答案。中文是给自己看的理解，English 是面试说的版本。

---

## 1. 三路检索各自解决什么问题，各自在什么情况下失败

### 中文

| 检索器 | 擅长 | 失败在 |
| --- | --- | --- |
| **向量** | 语义改写。查询和答案零词重叠也能匹配 | 罕见的精确 token。`work_mem`、`pg_stat_activity`、错误码——这些在 embedding 训练数据里出现太少，模型没有稳定表示 |
| **词法** | 词干化后的散文匹配。`updates` 能找到 `update` | 零词重叠时完全失效。而且**看不见标识符**——Postgres 分词器在下划线处断词 |
| **标识符** | snake_case 完整匹配 | 查询里没有标识符时返回空（大部分自然语言问句都是） |

**实测数字支撑第一行：**

| 配对 | 余弦相似度 |
| --- | --- |
| `"cancel my subscription"` vs `"how to terminate your plan"` | ~0.54 |
| `"work_mem"` vs `"memory allocated per sort operation"` | ~0.29 |
| `"error 42P01"` vs `"relation does not exist"` | ~0.17 |

语义改写 0.54，标识符 0.29，错误码 0.17——明明说的是同一件事，分数却差三倍。

**关键点：三者的失败模式互补。** 这才是融合的理由，不是「多几个检索器结果更好」。

候选集重叠度量化了这一点。纯改写查询 `why does the database slow down after many updates`：向量和词法的 top-20 **零重叠**，union 40 个。两个检索器找到了完全不同的东西。

### English

They fail in opposite directions, which is the actual argument for fusing
them — not that more retrievers give better results.

Dense vector search handles paraphrase: a query sharing no words with the
answer still matches, because the comparison happens in semantic space. It's
unreliable on rare exact tokens — identifiers, error codes — because those
appear too rarely in embedding training data for the model to hold a stable
representation. I measured it: a paraphrase pair scores around 0.54, while
`"error 42P01"` against `"relation does not exist"` scores 0.17, despite being
the same thing.

Lexical search is the inverse. It nails literal matches and fails completely
on zero word overlap. On this corpus it had a second failure I didn't
anticipate — it can't see identifiers at all, which is why there's a third
retriever.

The overlap number is the useful diagnostic. On a pure-paraphrase query, the
vector and lexical top-20 had zero chunks in common, 40 in the union. They
found entirely disjoint sets — that's the case fusion exists for. On an
identifier query the overlap was 8 to 10, so fusion was mostly reordering the
same candidates.

---

## 2. RRF 的公式是什么，为什么不用加权融合

### 中文

```
score(doc) = Σ  1 / (k + rank_in_retriever_i)
```

某个 chunk 在向量列表排第 2、词法列表排第 5：

```
1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151
```

只在向量列表排第 1：

```
1/(60+1) = 0.01639
```

**两个中等信号互相认可，赢过一个单独的最强信号。** 某个列表里没出现就不加那一项，没有惩罚项——这也是为什么从两路扩到三路不用改公式。

**加权融合的三个问题：**

```
final = 0.7 × cosine + 0.3 × ts_rank        # 不要这样
```

**量纲不兼容。** 余弦相似度在 [0, 1]，`ts_rank` 无上界，标识符检索的 `matched` 是小整数。直接加权等于让范围最大的那个主导。

**归一化不稳定。** min-max 归一化的基准取决于**当前这批结果**。同一个 chunk 在不同结果集里归一化后分数不同，跨查询不可比。

**分布形状不同。** 余弦相似度挤在窄区间——这个语料上 top-5 是 0.50 到 0.67；`ts_rank` 是长尾的。都归一化到 [0,1] 之后拉伸程度不一样，权重的含义就变形了。

RRF 只用排名，三个问题全绕过。超参从「两个权重要联调」变成一个 `k`。

### English

`score = Σ 1/(k + rank)` over retrievers. A chunk ranked 2nd and 5th scores
1/62 + 1/65 ≈ 0.0315, beating a chunk ranked 1st by one retriever alone at
1/61 ≈ 0.0164. Two moderate signals that agree outrank one strong signal
standing by itself.

Weighted score fusion requires normalizing scores that live on incompatible
scales — cosine is bounded in [0,1], `ts_rank` has no upper bound, and my
identifier retriever returns a small integer count. Normalization is the real
problem: min-max depends on the current result set, so the same chunk gets a
different normalized score depending on what else came back. Scores stop being
comparable across queries.

The distributions also have different shapes. Cosine similarities cluster
tightly — 0.50 to 0.67 across the top-5 on my corpus — while `ts_rank` is
long-tailed. Normalizing both to [0,1] stretches them differently, so the
weights don't mean what they appear to.

RRF sidesteps all of it by using only rank position. One hyperparameter instead
of two weights tuned jointly — and going from two retrievers to three needed no
change to the formula at all, because a retriever that returns nothing simply
contributes nothing.

---

## 3. `k = 60` 改成 `k = 1` 会发生什么

### 中文

`k` 控制**排名位置**相对于**是否出现**的重要性。

| k | 第 1 名得分 | 第 2 名得分 | 差距 |
| --- | --- | --- | --- |
| 1 | 1/2 = 0.500 | 1/3 = 0.333 | 巨大 |
| 60 | 1/61 = 0.0164 | 1/62 = 0.0161 | 几乎没有 |

**k = 60：** 头部被压平，所以「在多个列表里出现」比「在某个列表排第几」重要。这是我们想要的行为。

**k = 1：** 第 1 名的 0.5 谁都追不上——两个列表都排第 2 的 chunk 得 `1/3 + 1/3 = 0.67`，勉强超过；两个列表都排第 3 的得 `1/4 + 1/4 = 0.5`，打平。融合退化成「某个检索器排第一的那个」。

**换句话说：k 越小，RRF 越接近「取并集然后按单一最高排名排序」，融合的意义就消失了。**

### English

`k` controls how much rank position matters relative to mere presence.

At k=60, rank 1 scores 0.0164 and rank 2 scores 0.0161 — nearly identical. The
head of each list is flattened, so appearing in several lists matters more than
placing first in one. That's the behaviour worth having.

At k=1, rank 1 scores 0.5 and rank 2 scores 0.33. A chunk ranked 2nd by both
retrievers gets 0.67, barely beating a single 1st place; ranked 3rd by both it
gets 0.5, a tie. Fusion collapses into "whatever one retriever happened to put
first."

So smaller k moves RRF toward a plain union sorted by best single rank, which
defeats the point.

---

## 4. `work_mem` 在哪一步被切成 `work` + `mem`，为什么换配置没用

### 中文

**位置：** Step 3 入库，`content_tsv` 这个生成列。

```sql
content_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', content)      -- ← 这里切的
) STORED
```

`to_tsvector` 内部两个阶段：

```
原文      "work_mem controls memory"
             ↓  阶段 1：parser 分词
tokens    [work] [_] [mem] [controls] [memory]
             ↓  阶段 2：dictionary 词干化 + 去停用词
lexemes   'work' 'mem' 'control' 'memori'
```

**下划线在阶段 1 被当分隔符。这是 parser 的行为，不是词典的行为。**

配置（`'english'` / `'simple'`）只控制阶段 2。所以换成 `'simple'` 也一样切：

```sql
SELECT to_tsvector('simple', 'work_mem and pg_stat_activity');
-- 'activity':6 'and':3 'mem':2 'pg':4 'stat':5 'work':1
```

注意 `activity` 没被词干化、`and` 作为停用词保留了——说明 `simple` 确实关掉了阶段 2 的处理，但标识符照样断开。

**为什么损害比「失去精确性」更严重：**

```sql
SELECT to_tsvector('english', 'work_mem and pg_stat_activity and maintenance_work_mem');
-- 'activ':6 'mainten':8 'mem':2,10 'pg':4 'stat':5 'work':1,9
```

`'mem':2,10` — `work_mem` 和 `maintenance_work_mem` 都贡献了这个词元。两个参数共享三分之二的词元，词法上几乎无法区分。

而 `mem` 单独会匹配所有讲内存的段落——这正好是向量检索已经做得很好的事。

**结论：词法检索在这个语料上失去了唯一的比较优势。** 而语料就是按标识符密集选的。

**修法是绕开，不是修正。** `to_tsvector` 的分词行为没有配置项能改，只能另开一条通路——`identifiers TEXT[]` 列，用 Python 正则直接从 `content` 抽。

**这个 bug 是无声的：** `verify()` 检查 `content_tsv IS NULL` 是 0 行，列填好了、索引建了、查询有结果。结果只是错的。结构性检查通过，只有读实际词元才能发现。

### English

It happens inside `to_tsvector`, at ingest, in the generated `content_tsv`
column.

`to_tsvector` runs two stages: a parser that tokenizes, then a dictionary that
stems and drops stop words. The underscore is treated as a separator in stage
one — that's parser behaviour, and text-search configurations only control
stage two. I checked `'simple'`, which does no stemming at all, and it splits
the identifier the same way.

The damage is worse than losing exactness. `maintenance_work_mem` indexes as
`mainten` + `work` + `mem`, sharing two of three lexemes with `work_mem`, so
the two parameters are nearly indistinguishable lexically. And `mem` on its own
matches every passage about memory — which is precisely what dense retrieval
already does well. The lexical retriever had lost its only comparative
advantage, on exactly the content the corpus was chosen to exercise.

The fix is a bypass rather than a correction: a `text[]` column populated by
regex at ingest, with a GIN index, queried by array overlap. There's no
configuration option for underscore handling, so the only route is a surface
that never goes through the tokenizer.

What I'd emphasize is that the bug was silent. The column populated, the index
built, queries returned results — they were just the wrong results. A
structural check passes; only reading the actual lexemes reveals it. It
surfaced two steps later in an unrelated diagnostic that happened to print the
query's lexemes.

---

## 5. df 过滤为什么能替代 IDF，代价是什么

### 中文

**IDF 做的事：** 给罕见词高权重，给高频词低权重。BM25 的评分里每个词的贡献乘上 `log(N/df)`，所以 `pg_locks`（出现在 4 个 chunk）的权重远高于 `queri`（出现在 500 个）。

**`ts_rank` 不做这件事。** 它只算词频和位置，四个查询词一视同仁。

**df 过滤做的事：** 查词频表，把 `ndoc > 语料 × 0.15` 的词直接扔掉。

```
'see' | 'queri' | 'block' | 'pg_lock'
   ↓  过滤
'block' | 'pg_lock'
```

**为什么这能替代：** 两者都在解决同一个问题——「常见词不该主导排序」。IDF 用连续权重，df 过滤用二元的留/弃。方向一致，粒度不同。

**代价三条：**

**信息损失。** 高频词不是完全没用。`'databas'` 被扔掉了，但「database slow down」这个查询里它确实提供了一点语境。IDF 会给它低权重而不是零。

**阈值是硬边界。** 0.15 这个值上下的词命运完全不同——14.9% 保留、15.1% 扔掉，而它们的实际区分度几乎一样。IDF 没有这种断崖。

**短查询可能被清空。** `"how do I set this up"` 全是高频词，过滤后一个不剩。所以需要兜底：`terms = kept or dropped`，全被过滤就退回用全部词元。这个兜底本身是个妥协——它让最需要过滤的查询恰好绕过了过滤。

**为什么还是选它：** 真正实现 BM25 要在 `ts_stat` 输出上重写评分函数，工作量翻倍，而收益是同一个想法的精细化版本。而且阈值成了评估能调的参数（0.10 / 0.15 / 0.25），这个位置比一个没调过的精确实现更好。

### English

IDF weights each term's contribution by its rarity — `log(N/df)` — so a term
in 0.3% of the corpus counts far more than one in 40%. `ts_rank` has no such
term; it scores on frequency and position, treating every query term equally.

Document-frequency filtering addresses the same problem with a cruder
instrument: look up each query lexeme's document frequency, drop the ones above
a threshold. Both are saying "common terms shouldn't dominate" — one with
continuous weights, one with a binary keep-or-drop.

Three costs. Information loss: a high-frequency term isn't worthless, and IDF
would down-weight it rather than discard it. A hard boundary: a term at 14.9%
survives and one at 15.1% is dropped, despite near-identical discriminative
power. And short queries can be emptied entirely — "how do I set this up" is
all common words — so there's a fallback to the unfiltered terms, which means
the queries most in need of filtering are the ones that bypass it.

I chose it because implementing BM25 properly means reimplementing the scorer
over `ts_stat` output — roughly double the work for a refinement of the same
idea — and because the threshold becomes a parameter the eval harness tunes
rather than a number I guessed at. Being able to state the distinction
accurately matters more than the sophistication: "`ts_rank` is TF-IDF-family
with no IDF term, and I substituted document-frequency filtering for the
missing half" is true. "I implemented BM25" would not be.

---

## 6. 为什么 RRF 会放大噪音

### 中文

**实测的退化：** 查询 `how do I see which queries are blocked by pg_locks`

| 检索方式 | 正确答案的排名 |
| --- | --- |
| 纯向量 | **第 1** |
| RRF 融合 | **第 4** |

融合让结果变差了。

**机制：**

1. 查询分词成 `'see' | 'queri' | 'block' | 'pg_lock'`
2. `queri` 出现在 40% 的 chunk 里，`ts_rank` 没有 IDF，所以它和 `pg_lock` 同等重要
3. 词法检索的 top-5 填满了「只是经常提到 query」的章节，其中包括一个 distractor
4. 那个 distractor 在向量列表里也有（排第 3，语义上确实沾边）
5. **RRF 看到它在两个列表都出现，给了跨检索器一致性加分**，推到第 1

**核心：RRF 奖励一致性，而一致性只有在两个检索器都在排信号时才是证据。**

如果一个检索器在输出噪音，「两边都认可」就不是证据而是巧合。融合不过滤噪音——它放大输入所一致的东西。

**这个观察的价值：** 所有教程讲 RRF 都在讲它的好处（不用归一化、鲁棒、超参少），没人讲它的前提。而这个前提在实际系统里经常不满足。

### English

I hit a measured regression: on one query, RRF ranked the correct chunk 4th
where plain vector search had it 1st.

The mechanism is worth walking through. The query tokenized to four terms, one
of which — the stem for "query" — appears in about 40% of a PostgreSQL corpus.
With no IDF weighting it counted as much as the rare identifier, so the lexical
top-5 filled with sections that merely mention "query" often. One of those was
also in the vector results, ranked 3rd, because it was semantically adjacent.
RRF saw it in both lists and applied its cross-retriever agreement bonus,
promoting it to 1st.

**RRF rewards agreement, and agreement is only evidence when every retriever is
ranking on signal.** If one retriever is producing noise, "both lists contain
it" is a coincidence, not corroboration. Fusion doesn't filter noise — it
amplifies whatever the inputs agree on.

Every treatment of RRF I've read covers its advantages — no score
normalization, robust to scale differences, one hyperparameter. None of them
state the precondition, and the precondition fails in practice more often than
you'd expect.

---

## 7. section cap 解决什么问题，为什么 rerank 解决不了

### 中文

**问题：** 查询 `preventing transaction id wraparound` 的 top-5

```
1  24.1.5. Preventing Transaction ID Wraparound Failures
2  24.1.5. Preventing Transaction ID Wraparound Failures
3  24.1.5. Preventing Transaction ID Wraparound Failures
4  24.1.5. Preventing Transaction ID Wraparound Failures
5  24.1.5. Preventing Transaction ID Wraparound Failures
```

同一个 section 被切成五片以上，全部占据 top-5。

**两个后果：**

- **有效多样性只有一个 section 宽**，但看起来是五条结果深
- **RRF 在这里毫无作用** —— 两个检索器的候选集是同一批内容，没有东西可融合

**为什么 rerank 解决不了：**

rerank 的工作是**判断相关性**。而这五片**确实都相关**——它们是同一个主题的连续段落。cross-encoder 会给它们全都打高分，垄断原封不动。

rerank 修的是「候选池里有相关的也有不相关的，排序不对」。这里的问题是「候选池里全是相关的，但来自同一个地方」。这是**多样性问题**，不是**相关性问题**。

**所以必须在融合时结构性处理：**

```python
MAX_PER_SECTION = 2
```

加 cap 之后：

```
1  24.1.5. Preventing Transaction ID Wraparound Failures
2  24.1.5. Preventing Transaction ID Wraparound Failures
3  19.10. Automatic Vacuuming            ← 之前被挤掉了
4  13.2. Transaction Isolation
5  19.11.1. Statement Behavior
```

`19.10. Automatic Vacuuming` 是真的相关（autovacuum 就是防 wraparound 的机制），之前完全看不到。

**这个 cap 的代价：** 牺牲 section 内的召回，换 section 间的覆盖。

- 答案需要两个 section 拼起来 → cap 是对的
- 答案分散在一个 section 的五片里 → cap 在删正确答案

哪种情况更多是实证问题，所以留成参数让评估来定。

### English

On one query the top-5 was five chunks from the same section. Two
consequences: effective diversity was one section wide while appearing five
results deep, and RRF had nothing to contribute because both retrievers'
candidate sets were the same content.

Reranking can't fix this, and the reason is worth being precise about. A
reranker judges relevance. Those five chunks genuinely *are* relevant — they're
consecutive passages on the same topic — so a cross-encoder scores them all
highly and the monopoly survives intact. Reranking fixes "the candidate pool
has relevant and irrelevant chunks mixed, in the wrong order." This is "the
candidate pool is entirely relevant, but all from one place." A diversity
problem, not a relevance problem.

So it has to be handled structurally, at fusion time — a cap of two chunks per
section. After the cap, a section on automatic vacuuming appeared at rank 3,
genuinely relevant and previously invisible.

The cap trades recall within a section for coverage across sections. That's
right when an answer needs two sections combined and wrong when it's spread
across five chunks of one. Which case dominates is empirical, so it's a
parameter the eval harness decides rather than something I fixed by judgment.

---

## 8. identifier 检索的 tiebreak 第一版为什么是错的

### 中文

**第一版的排序：**

```sql
ORDER BY matched DESC, cardinality(identifiers) ASC
```

- `matched` = chunk 命中了几个查询里的标识符
- `cardinality(identifiers)` = chunk 一共有多少个标识符

**当时的理由：** 偏好「专注」的 chunk。专门讲 `work_mem` 的小节应该赢过列了四十个参数的汇总表格。

**为什么错：**

大部分查询只有**一个**标识符。所以所有命中的 chunk 在 `matched` 上全部打平（都是 1），**排序完全由 tiebreak 决定**。

于是实际效果变成「标识符总数越少排越前」——也就是「只提一嘴的 chunk 赢」。

**实测：** 查询 `pg_locks`

```
1 D  13.2.3. Serializable Isolation Level    ← 顺带提了 pg_locks 一次
2 D  13.3. Explicit Locking
3    27.3. Viewing Locks                     ← 专门讲这个视图的章节，排第 3
```

专门讲 `pg_locks` 的章节自然也会提到 `pg_stat_activity`、`pg_blocking_pids` 等相关视图，所以它的标识符总数多，反而被惩罚了。

**修法：按提及频次排序**

```sql
ORDER BY matched DESC, mentions DESC

-- mentions:
COALESCE((
    SELECT SUM(regexp_count(content, q.ident, 1, 'i'))
    FROM unnest($1::text[]) AS q(ident)
), 0)
```

专门讲 `pg_locks` 的章节会提它十几次；顺带引用的提一两次。**这个信号才真的能分开它们。**

修完之后 `work_mem` 查询的 identifier top-5 全是 eval chunk，之前混着 `18.4.4. Linux Memory Overcommit`——一个只提了 `work_mem` 一次的章节。

**教训：** tiebreak 在「主排序键经常打平」的情况下就是主排序键。设计它时要想清楚它实际在排什么，而不是它听起来在排什么。

### English

The first version broke ties by total identifier count ascending, reasoning
that a focused chunk should beat a summary table listing forty parameters.

The flaw is that most queries carry a single identifier, so every match ties on
the primary key and the tiebreak decides the entire ordering. What it actually
ranked was "fewest identifiers wins" — which means "mentions it in passing
wins."

Concretely, for `pg_locks`: a section on serializable isolation that cites the
view once ranked 1st, while the section documenting the view ranked 3rd. The
documenting section naturally also mentions `pg_stat_activity` and
`pg_blocking_pids`, so it carried more identifiers and was penalized for it.

The fix ranks by mention frequency instead — how many times the matched
identifiers actually appear in the text. A section about `pg_locks` names it a
dozen times; a section citing it names it once. After the change, the
identifier top-5 for `work_mem` was entirely eval chunks, where it previously
included a section that mentions `work_mem` exactly once.

The general lesson: when the primary sort key ties often, the tiebreak *is* the
sort key. It needs to be designed for what it actually orders, not for what it
sounds like it orders.

### 补充：tiebreak 第二次也错了 / Follow-up: wrong a second time

**中文**

改成「按提及次数」之后还有一个 bug：计数用的是子串匹配。`work_mem` 是 `maintenance_work_mem` 和 `autovacuum_work_mem` 的子串，所以讲那两个参数的 chunk 被多算了。chunk 1032 里 `work_mem` 子串出现 5 次，整词只有 1 次。

修法是按词边界计数：PostgreSQL 正则里 `\m` `\M` 是词首词尾，下划线算词内字符，所以 `\mwork_mem\M` 不会匹配 `maintenance_work_mem` 里面那段。

和第 4 题是同一类错误：分词器把标识符当成一串单词，tiebreak 把它当成一串字符。**标识符是一个整体，每一层碰到它都要当整体处理。**

**English**

The mention count was a substring count, and `work_mem` is a substring of
`maintenance_work_mem` and `autovacuum_work_mem`. One chunk was credited with
five mentions for a term it uses once as a word. Counting on word boundaries —
`\m` and `\M` in Postgres regex, where underscore is a word character — fixes
it. It's the same failure class as the tokenizer split: an identifier has to be
treated as a unit at every layer that touches it, and I found two layers that
didn't.

---

## 9. section cap 应该放在 rerank 之前还是之后

### 中文

**原来是 cap → rerank，现在改成 rerank → cap。**

cap 要决定「一个 section 留哪几片」。放在 rerank 之前，它只能按 RRF 排名来决定——而同 section 的几片 chunk heading、section_id、URL 全部相同，RRF 排名恰恰是**最分不清它们的信号**。这正是需要 rerank 的原因。

所以原来的顺序等于：在唯一能区分它们的组件之前，按最不可靠的信号把候选扔掉了。

改成先 rerank，reranker 看到完整的 40 个候选按内容排好，cap 再在已知相关性的列表上做多样性裁剪。

**但这只是原理上的论证。** 新顺序是否实际更好，没有证据——两种顺序都保留成配置，交给评估决定。

还有一个更根本的问题：cap 假设「一个 section = 一个主题」。`24.1.5. Preventing Transaction ID Wraparound Failures` 符合；`19.4.1. Memory` 不符合——它是十几个独立参数的列表，切成 8 片。把它限制在 2 片，等于把十几个不同的答案当成了彼此的重复。

### English

It originally ran cap, then rerank. The cap decides which chunks of a section
survive, and before reranking the only signal it has is RRF rank — which,
within one section, is the least reliable signal available, because the chunks
share heading, section id, and URL. That's the whole reason reranking exists. So
the original order discarded candidates, using the one signal that can't tell
them apart, before the one component that can.

I reordered it to rerank first. But that's an argument, not a measurement, so
both orders are registered as eval configurations rather than the choice
resting on my reasoning.

There's a deeper problem with the cap on this corpus: it assumes a section is
one topic. A section like "Preventing Transaction ID Wraparound Failures" is.
"19.4.1. Memory" is a reference list of a dozen independent parameters across
eight chunks — capping it at two treats a dozen distinct answers as duplicates.

---

## 10. 怎么知道一个答案需要两个 chunk

### 中文

有两种情况看起来都是「答案涉及两个 chunk」，但意思相反：

| 每片单独能答吗 | 情况 | 算分规则 |
| --- | --- | --- |
| 能，好几片都能 | 重复（overlap 导致） | 命中任意一片就算对（`any`） |
| 都不能，合起来能 | 真多跳 | 要全部命中（`all`），可以给部分分 |

**判断方法只有读。** 对每个候选 chunk 问：如果模型只看到这一片，能回答吗？

**Cohere rerank 做不了这个判断。** 它对每个 (问题, chunk) 独立打分，衡量的是**相关性**，不是**充分性**。多跳题的两半各自只是「有点相关」，分数上看不出它们是互补的一对。

现在这一步交给 LLM judge 做（`06_build_golden.py`）：每片单独问一次「只看这一片能不能完整回答」，得到 yes / partial / no；两家厂商的 judge 各判一次，取较弱的那个。只有当一道题没有 yes、恰好有 2–3 片 partial 时，才把这几片放在一起再问一次「合起来能不能完整回答、哪几片是必需的」，两个 judge 必须给出同一组才记为 `all`，否则丢弃这道题。

judge 本身靠两样东西保证可信：跑之前必须通过手工核对过的校准题，跑完报告两个 judge 的一致率（Cohen's kappa）。golden dataset 是标准本身，所以判断器要先被验证，而不是逐题人工确认。

**我自己就判断错过一次：** 以为「调大 maintenance_work_mem 会给 autovacuum worker 更多内存吗」是多跳题，需要 1031 + 1032。读了 1032 原文——*"It defaults to -1, indicating that the value of maintenance_work_mem should be used instead"*——单独一片就能回答。

**还有一次更关键的误判：** 我用「chunk 1030 在不在 top-5」来测 section cap 有没有删掉正确答案，结果报告删了。实际上 `work_mem` 那段描述跨过了 1030 和 1031 的边界，64 token overlap 让关键句同时出现在两片里——1031 一直在 top-5 里。**系统是对的，是只认一个 id 的检查错了。**

### English

Two situations both look like "the answer involves two chunks" and mean
opposite things. If each chunk alone can answer — usually because overlap put
the same sentence in both — any one of them in the top-5 counts. If neither
alone can answer but together they can, that's true multi-hop, and every piece
has to be retrieved.

The only way to tell is to read each candidate and ask whether it would suffice
on its own. In this project LLM judges from two vendors do that reading, one
passage per call, and only when a question has two or three partial answers do
they see those passages together and name which ones are required. The judges
are trusted because they pass hand-verified calibration cases first, not
because each label is checked by hand. A reranker can't make that call: it scores each passage
independently, so it measures relevance, not sufficiency. Two halves of a
multi-hop answer each look moderately relevant, and nothing in the scores says
they complete each other.

I learned this by getting it wrong. I tested whether a diversity cap was
deleting correct answers by checking for one specific chunk id in the top-5,
and the check said yes. But the answering sentence straddled a chunk boundary
and, through the overlap, appeared in two chunks — the other one was in the
top-5 the whole time. The system was right and my check was wrong, because it
recognized one correct chunk and there were two. That's why the golden dataset
stores ground truth as a set with an explicit any-or-all match rule.

---

## 11. 查询为什么会被词干化两次

### 中文

词法检索先用 `to_tsvector('english')` 把问题转成词元，这一步是对的，和索引用的是同一套处理。但之后把这些词元又送进了 `to_tsquery('english', ...)`，**又被词干化了一次**。

问题在于 Snowball 英文词干算法**不是幂等的**：

| 原词 | 第一次（和索引一致） | 第二次（查询实际用的） |
| --- | --- | --- |
| database | databas | databa |
| increase | increas | increa |
| release | releas | relea |

以「元音 + s」结尾的词干，第二次会再丢掉一个 s。索引里存的是 `databas`，查询找的是 `databa`，永远匹配不上。

在本地 PostgreSQL 上实测：查 `database`，旧写法匹配 **0 条**，新写法匹配 3 条。

修法：词元已经是索引的形式，就不该再过一遍分词器。把它们拼成 tsquery 字面量，用类型转换（`$1::tsquery`）而不是 `to_tsquery`。

和下划线分词是同一类问题：**同一个值跨过两层处理，而两层对文本的归一化方式不同。** 规则是：只归一化一次，跨层传递时用对方不会重新解释的形式。

### English

The lexical path stemmed query terms once with `to_tsvector` — correct, same
configuration as the index — and then passed those lexemes to
`to_tsquery('english', ...)`, which stems them again. Snowball English isn't
idempotent: `database` becomes `databas`, which matches the index, and then
`databa`, which matches nothing. Any stem ending in a vowel plus "s" loses the
"s" on a second pass — database, increase, release, decrease, all common words
in a database manual.

I confirmed it against a real Postgres instance: zero matches for "database"
before the fix, three after. The fix is to build the tsquery as a literal and
cast it, so already-normalized lexemes reach the index untouched.

It's the same shape as the underscore bug: a value crossing a boundary between
two layers that normalize text differently. The rule I took from both is to
normalize exactly once, and to pass normalized values across boundaries in a
form the other side won't reinterpret.

---

## 12. eval 章节是怎么选出来的

### 中文

选了五章：Server Configuration、Indexes、Monitoring、Performance Tips、Routine Maintenance。三个标准，外加一个事后才发现的前提。

**标准 1：标识符密集，语料要能暴露 hybrid 想解决的失败模式。**

这五章里全是 `work_mem`、`pg_stat_activity`、`autovacuum_vacuum_scale_factor` 这类参数名和视图名，而这正是向量检索的短板——实测语义改写的相似度 0.54，`work_mem` 和它的定义只有 0.29。

如果语料全是散文，词法和向量检索结果几乎一样，RRF 融合什么也贡献不了，整个架构就没法证明自己有价值。

**标准 2：这个领域我能判断结果对不对。**

ground truth 虽然由 LLM judge 标注，但 judge 要先通过我手工核对过的校准题，跑完还会抽一批题给人看。这两步都要求我自己知道答案。换成 C 扩展接口或者排序规则那几章，给我五个看起来都合理的 chunk，我分不出哪个是对的，校准和抽查就都没有意义。

干扰章节对这个条件要求低得多：不从里面出题，也不从里面取校准题和抽查题，judge 只在它们进了某道题的候选池时才标。所以完全不熟的 `functions.html` 可以放心当干扰项，而且它词汇量大，干扰效果好。

**标准 3：规模要让题目集有覆盖度。**

几十道题摊在 5 章上够用，摊在全部 21 章上就太稀薄，题数和 judge 成本都得翻几倍。而且规模不是这个项目要证明的东西。

**事后发现的前提：** 按标识符密集选语料，隐含假设了「词法检索能完整匹配标识符」。实际上 PostgreSQL 分词器在下划线处断词，`work_mem` 变成 `work` + `mem`，这个能力根本不存在。加了第三路标识符检索才补上。

**没有那一路，标准 1 的理由理论成立、实际空转**——语料暴露了一个系统处理不了的失败模式，跑出来的数字会显示「hybrid 在这里没什么用」，但原因是错的。

### English

Three criteria, plus a condition I discovered late.

First, identifier density: the corpus has to exercise the failure mode hybrid
retrieval exists to fix. These chapters are saturated with parameter and view
names, and that's exactly where dense retrieval is weak — I measured a
paraphrase pair at 0.54 cosine against 0.29 for `work_mem` versus its own
definition. On a corpus of pure prose, lexical and vector search return nearly
the same results, RRF has nothing to contribute, and the architecture would
look sophisticated while demonstrating nothing.

Second, it had to be a domain I can judge. The ground truth is labelled by
LLM judges, but I only trust them after they pass calibration cases I verified
by hand, and I spot-check a sample of the accepted questions. Both need me to
know the answer. Show me five plausible chunks about the C extension interface
and I couldn't tell you which is correct, so neither check would mean anything.
Distractor chapters need that property much less: no question, calibration
case, or spot check comes from them, which is why something completely outside
my knowledge works well there.

Third, scope. A few dozen questions across five chapters gives real coverage;
spread across all twenty-one it would be too thin to trust.

The condition I missed: choosing the corpus for identifier density assumes the
lexical retriever can match an identifier whole. It can't — Postgres splits on
underscores before any dictionary runs. I had to add a third retriever over a
separate identifier column. Without it the corpus rationale would have been
sound in theory and inert in practice: the corpus would exercise a failure mode
the system had no machinery to handle, and the numbers would have said "hybrid
doesn't help much here" for entirely the wrong reason.

That one is worth saying out loud in an interview, because the reasoning was
right and the outcome was still nearly wrong. A corpus choice justified by a
capability is only as good as the check that the capability exists.

---

## 13. 同一个 section 的几片 chunk 内容不一样，前三路检索器为什么分不出来

### 中文

因为**三路检索器都不读内容**，它们比的是内容的某种压缩表示。

**向量检索**：把一整段压成 1536 个数字再比。同一 section 的两片都会被压成「这段在讲内存配置」，查询也落在同一片区域。压缩是有损的——「按操作算不按连接算」这个具体事实在 1536 维里只是一个很微弱的方向，被「在讲内存配置」这个强得多的信号淹没。实测两片的分差可能只有 0.02。

**词法检索**：比词元集合。同 section 的几片都含 `mem`、`memori`、`set`、`configur`，重合度极高。`ts_rank` 按词频算分，分不出谁真的回答了问题。

**标识符检索**：比标识符数组。两片都含 `work_mem`，`matched` 都等于 1，只能靠提及次数这种间接信号。

**cross-encoder 跳过了压缩这一步**，把查询和整段原文拼在一起送进模型：

```
"Is work_mem a per-connection limit? [SEP] ...each operation generally being
 allowed to use as much memory as this value specifies. Also, several running
 sessions could be doing such operations concurrently..."
```

attention 跨越拼接点，查询里的 "per-connection" 直接对上原文里的 "several running sessions"。这是**阅读**，不是算相似度。

类比：前三路是靠目录和索引找书——知道这本书讲内存配置，但不知道第 47 页写了什么。cross-encoder 是把那一页读完再判断。准，代价是不能预先算好。

### English

They differ in content, but none of the three retrievers reads content — each
compares a compressed representation.

Vector search compresses a passage to 1,536 numbers. Two chunks from the same
section both reduce to "this is about memory configuration", and so does the
query. Compression is lossy: the specific fact that a limit is per-operation
rather than per-connection survives as a faint direction in the vector, swamped
by the topical signal. The gap between an answering chunk and a non-answering
one can be 0.02.

Lexical search compares lexeme sets, and same-section chunks share nearly all
of them. Identifier search compares identifier arrays, and both chunks contain
the identifier, so both score 1 and the ranking falls to a proxy.

A cross-encoder skips the compression. Query and passage go in as one input, so
attention runs across the join — "per-connection" in the question attends
directly to "several running sessions" in the passage. It's reading
comprehension, not similarity.

The analogy I'd use: the first three find a book by its index and table of
contents. They know it's about memory configuration; they don't know what page
47 says. The cross-encoder reads page 47. That's why it's accurate and why
nothing can be precomputed.

---

## 14. 怎么知道一个问题需不需要拆成子问题再查

### 中文

三种做法：

| 做法 | 成本 | 局限 |
| --- | --- | --- |
| 总是拆 | 每个查询多一次 LLM 调用 | 大部分问题是单点的，拆出来还是一个子问题，白花 |
| 让 LLM 判断要不要拆 | 每个查询多一次 LLM 调用 | 在看到任何证据之前预测难度 |
| 用检索信号触发 | 免费，信号已经有了 | 需要一个阈值，而阈值要测出来 |

第三种可行是因为 rerank 已经产出了这个信号。实测 top-1 分数：覆盖好的查询 0.897，答案根本不在语料里的查询 0.405。

**分数低说明没有单片能完整回答**——可能是问题需要多个点，也可能是检索失败了。两种情况的处理一样（重新组织查询再查），所以不需要提前分辨。

它还天然覆盖了顺序多跳：第二个查询是**看过第一次检索结果之后**才写的，模型已经拿到了需要的那个词（比如「可见性映射」），不用在拆题时凭空猜。

**阈值存不存在是个实证问题，golden dataset 正好能回答**：把多跳题和单点题的 rerank top-1 分数画出来，看有没有一条线能分开。分不开就说明这个信号不够，触发条件得换成显式的 LLM 判断——那是一个结论，不是失败。

### English

Three options. Always decompose costs an LLM call per query and wastes it on
the majority of questions that are single-fact. Asking an LLM whether to
decompose costs the same call and predicts difficulty before seeing any
evidence. Triggering on a retrieval signal is free, because reranking already
produces one.

I have that signal: top-1 rerank scores ranged from 0.897 on a well-covered
query to 0.405 on one whose answer isn't in the corpus. A low score means no
single passage covers the question — either it needs several facts or retrieval
failed. Both call for the same response, so they don't need to be told apart.

It also handles sequential multi-hop naturally: the second query gets written
after seeing the first retrieval, so the model has the term it needs rather
than guessing it during up-front decomposition.

Whether a usable threshold exists is an empirical question, and the golden
dataset answers it — plot top-1 rerank score for the multi-hop questions
against the single-fact ones and see whether a line separates them. If it
doesn't, the trigger has to be an explicit LLM judgment instead. That's a
finding, not a failure.

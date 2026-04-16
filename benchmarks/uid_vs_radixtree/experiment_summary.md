# UID Lookup vs RadixTree Lookup 实验总结

## 0. 实验环境
独占一台完整的GPU机器

```bash
salloc --partition="a100-pcie-40gb@cr+mp/x11spgtf/1gpu-24cpu-32gb"        --gres=gpu:1 --cpus-per-task=8 --mem=24G        --time=08:00:00 --job-name=lyl_bench --no-shell
```

## 1. 实验目的

在recsys-example中，如果将原本基于 `uid -> cache entry` 的直接查找方式，替换为 FlexKV 中的 RadixTree 前缀匹配方式，那么**额外查找开销到底有多大**。

主要包括下面几个关键点：

1. **绝对开销**：UID 查找和 RadixTree 查找分别耗时多少。
2. **开销来源**：额外成本主要来自哈希生成、树遍历，还是 Python 调用路径。
3. **参数敏感性**：开销如何随 cache 大小、序列长度、batch size、命中率变化。

从设计上看，这个实验对应的是如下对比：

- **Baseline**：recsys-examples 风格的 UID 独占查找，本质是一次 `unordered_map.find(uid)`。
- **Treatment**：FlexKV 风格的 RadixTree 前缀查找，本质是 `token ids -> block hashes -> match_prefix()`。

值得注意的是，本实验比较的是**两种索引思路**的开销差异，而不是把 uid 作为 RadixTree key 的特殊实现。uid 作为树节点前缀没有语义价值，因此不在实验范围内。

## 2. 实验代码设计

本实验共使用两份核心数据文件和两类 benchmark 程序：

- [benchmark_uid_vs_radixtree.py](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/benchmark_uid_vs_radixtree.py)
- [uid_vs_radixtree_bench_stl.cpp](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree_bench_stl.cpp)
<!-- - [uid_vs_radixtree.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree.csv) -->
- [uid_vs_radixtree_large_scale.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree_large_scale.csv)
- [stl.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/stl.csv)

### 2.1 Python benchmark：测当前 Python 接入路径下的总开销

Python 版本使用的是 [benchmark_uid_vs_radixtree.py](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/benchmark_uid_vs_radixtree.py)。它直接调用 FlexKV 的 Python/C-extension 接口：

- `CRadixTreeIndex`： FlexKV RadixTree 实现
- `SequenceMeta`： block hash 生成逻辑

这个脚本会测四类时间：

1. `uid_us`
   含义：Python `dict.get(uid)` 的 batch 查找时间。
   作用：测 recsys-examples 的 UID O(1) baseline。

2. `hash_us`
   含义：`SequenceMeta(token_ids, tokens_per_block)` 的构造时间。
   作用：测 token 序列生成 block hash 的成本。

3. `tree_us`
   含义：`index.match_prefix(...)` 的调用时间。
   作用：测已有 hash 情况下的 RadixTree 查找成本。

4. `radix_total_us`
   含义：`hash_us + tree_us`。
   作用：表示当前 Python 热路径下，完整 RadixTree 查询的总代价。

它的参数矩阵包括：


- `sequence length`: （128） 256 / 1024 / 4096 / 16k        
- `blocks_per_entry`: 16 / 64 / 256            （32 / 64）         
- `batch_size`: 1 / 8 / 32 / 128.              (1 + 多batch)
- `overlapping ratio`: 0.0 / 0.5 / 1.0
- `cache_size`: 



1. 图 每个固定的batch会有一个 （随着sequence length）：绝对时间 + ratio （uid直接用 python）
2. 潜在问题： cache 情况，



- [uid_vs_radixtree.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree.csv) 是 quick 模式输出，只覆盖较小参数子集，用于快速验证环境。
- [uid_vs_radixtree_large_scale.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree_large_scale.csv) 是 full 模式输出，覆盖完整参数矩阵，适合作为正式结论依据。

### 2.2 STL benchmark：测 RadixTree 算法本身的开销

由于真实 FlexKV C++ benchmark 在当前容器环境下编译受阻，我们额外实现了一个**纯 STL 版本**：[uid_vs_radixtree_bench_stl.cpp](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree_bench_stl.cpp)。

它复刻了以下数据结构：

- 每个节点维护一段 `block_hashes`
- 每个节点维护 `children` 哈希表
- 查找逻辑采用“边界 child 匹配 + 节点内二分查找”的方式

因此 STL benchmark 的结果代表的是：

**如果只看 RadixTree 作为一种数据结构，相比 UID 直接哈希查找，额外开销到底有多大。**

结果写入 [stl.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/stl.csv)。

### 2.3 为什么要同时保留 Python 和 STL 两个版本

两者回答的问题不同：

- **STL 版本**回答：算法本身慢多少。可以提供一个下界值：也就是说纯算法实现的成本到底是多少？
- **Python 版本**回答：当前 patch 路径下，实际热路径会慢多少。可以提供一个上界值：当前的实现路径的成本。


## 3. 实验结果

### 3.1 STL 结果：算法本身开销中等，不是灾难性的

从 [stl.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/stl.csv) 可以看到：

- RadixTree 相比 UID 查找的 overhead 大约为 **1.1x 到 16.5x**。
- 大多数配置下，overhead 集中在 **2x 到 8x** 左右。
- 随着 `cache_size`、`blocks_per_entry`、`batch_size` 增大，overhead 会增加。
- `hit_rate = 100%` 时通常更慢，因为树匹配更深。

这说明：

**RadixTree 作为数据结构本身是有额外开销的**

### 3.2 Python 结果：总代价被显著放大

从 [uid_vs_radixtree_large_scale.csv](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree_large_scale.csv) 可以看到：

- Python 版本的 overhead 大约为 **30.6x 到 478.8x**。
- 这个量级显著高于 STL 版本。

结合 `hash_us`、`tree_us`、`radix_total_us` 的拆分可以判断：

- 真正显著增加的不是树本身，而是 Python 路径带来的管理成本。
- 这些成本包括：
  - `SequenceMeta` 构造
  - block hash 生成
  - numpy / torch 转换
  - Python 到 C-extension 的调用边界
  - 每次查询都创建临时对象

因此 Python 结果更适合解释为：

**按当前 Python patch 接入方式，完整查询路径会被显著放大。**

### 3.3 结论

综合 STL 与 Python 结果，可以得出如下判断：

1. **radix本身是具有竞争力的**
   即使我们从结果中看到，radix的结果比普通的大很多，但是这是因为Python 结果里混入了大量非算法成本。

2. **可以把 STL 结果视为算法成本下界。**
   即使完全不考虑 Python，RadixTree 查找也确实比 UID O(1) 查找更贵，但这种更贵仍在可分析、可接受的范围内。

3. **可以把 Python large-scale 结果视为当前实现路径上界。**
   如果继续沿着当前 Python 热路径接入，线上延迟会被明显放大。

4. **结论：**
   - 若后续继续推进 FlexKV / RadixTree 方向，应尽量把 hash + lookup 逻辑压到 C++ 热路径中。
   - 若必须长期依赖当前 Python 热路径，则不建议直接用于强延迟约束场景。

## 4. 实验的不足

虽然本实验已经足够支持方向判断，但仍然存在以下限制。

### 4.1 没有拿到“真实 FlexKV C++ API benchmark”的最终结果

原始设计里有 [uid_vs_radixtree_bench.cpp](/home/scratch.noliu_gpu/FlexKV/benchmarks/uid_vs_radixtree/uid_vs_radixtree_bench.cpp)，目的是直接调用真实 `CRadixTreeIndex::match_prefix()` 来测 C++ 层开销。

但由于当前容器环境下 FlexKV 源码与本地编译环境不兼容，这个 benchmark 没有得到可用结果。

因此当前结论依赖的是：

- STL 版：算法层
- Python 版：当前接入路径层
- mode prebuilt版：对sequencemeta的重复构造变成一次构造，减少对hash表的重复构造。

中间少了一层“真实 FlexKV C++ API 层”的直接数据。


### 4.2 当前 workload 仍是合成 workload

实验里使用的是随机生成的 token 序列和受控命中率，而不是实际推荐请求日志。而且 hit rate 设置的也是按步长设置的，所以如果有真实的数据分布可能会更好。

### 4.3 Python benchmark 会高估当前 patch 方案的热路径代价

虽然 Python 路径成本在你们当前 patch 设计中确实存在，但 benchmark 的实现方式仍然比线上实际系统更“保守”，因为它在每个 batch 中频繁构造 `SequenceMeta`、`torch.from_numpy(...)`、临时对象等。

因此它更适合作为：

- 当前实现路径的上界

而不是未来生产实现的精确估计。

## 5 下一步计划

当前实验仅仅关注前缀复用的能力，仅仅关注 使用radix的前缀复用会带来的额外开销（包括：hash构建+radix查找的时间）

所以我们接下来可以测试：
- 前缀复用可以节省多少的计算（prefill）
- 可以节省多少 GPU 的显存
- 可以节省多少 CPU/SSD KV 传输
- 端到端的 latency/throught 是否更优 （与之前的相似）
- 真实的工作负载的讨论
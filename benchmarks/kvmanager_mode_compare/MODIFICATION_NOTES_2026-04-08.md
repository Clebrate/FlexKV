# KVManager Mode Compare 变更记录（2026-04-08）

## 背景

在 `direct` / `server_client` 对比过程中，出现了两类问题：

1. `run_compare.py` 在某些场景会卡住（子进程输出管道 EOF 行为不稳定）。
2. 多轮运行时，后续轮次可能出现 `cudaHostRegister failed with error code 1`，怀疑与资源回收不充分有关。

本次改动分为三块：**benchmark 侧增强**、**FlexKV worker 侧回收增强**、**KVServer 侧回收修复**。

---

## 本次改动清单

## 1) `benchmarks/kvmanager_mode_compare/run_compare.py`

### 新增能力

- 支持多轮统计：
  - `--warmup`：预热轮数（不计入结果）
  - `--repeats`：统计轮数
  - `--summary-output`：输出聚合结果 JSON
- 多轮时交替模式顺序（减少时序漂移偏差）：
  - 偶数轮：`direct -> server_client`
  - 奇数轮：`server_client -> direct`
- 新增聚合统计输出：
  - `mean / median / p95 / min / max / std`
  - 并打印 `server_client - direct` 的均值差值

### 稳定性改动

- 子进程输出不再通过 `stdout=PIPE` 直接抓取；
- 改为写入临时文件后再解析 `FLEXKV_BENCH_JSON`，规避子孙进程继承 fd 导致 `communicate()` 卡住的问题。

### 默认配置修正

- `--config` 默认值恢复为：
  - `benchmarks/kvmanager_mode_compare/smoke_config.yml`

---

## 2) `benchmarks/kvmanager_mode_compare/formal_lite_config.yml`

新增中等规模配置（介于 smoke 与 example 之间），用于在受限容器中提升可跑性：

- `num_layers: 16`
- `cpu_cache_gb: 2`
- `ssd_cache_gb: 4`

> 注意：在当前环境下，即使是 lite 配置，仍可能触发 `cudaHostRegister` 失败，和容器 memlock / pinned memory 额度相关。

---

## 3) `benchmarks/kvmanager_mode_compare/bench_one_run.py`

- 本轮最终状态：**保持原始等待行为**（未保留 fail-fast/timeout 防呆逻辑）。
- 也就是说 `is_ready()` 等待逻辑当前是原来的风格（会持续 heartbeat 日志）。

---

## 4) `flexkv/transfer/worker.py`（FlexKV 源码层）

为满足“每轮结束主动回收资源”的目标，增加了显式释放逻辑：

- `TransferWorkerBase.shutdown()`：
  - 显式 `cudaHostUnregister(op_buffer_tensor)`
  - 关闭 worker pipe
- `GPUCPUTransferWorker.shutdown()`：
  - 显式 `cudaHostUnregister(cpu_tensor)` 后调用 `super().shutdown()`
- `tpGPUCPUTransferWorker.shutdown()`：
  - 同上，显式解除 `cpu_tensor` 注册
- `PEER2CPUTransferWorker.shutdown()`：
  - 末尾补充 `super().shutdown()`
- `WorkerHandle.shutdown()`：
  - 优先优雅退出（`join` 等待更久）
  - 超时后才 `terminate/kill`
  - 最后 `process.close()`

此外修正了 tp worker 初始化中两个变量引用：

- `cpu_blocks_ptr = self.cpu_tensor.data_ptr()`
- `dp_group_id -> self.dp_group_id`

---

## 5) `flexkv/server/server.py`（FlexKV 源码层）

修复了 server 退出路径中的一个关键问题：

- 原逻辑在 `run()` 收尾阶段调用的是 `self.kvmanager.shutdown()`（该对象并不存在），
  导致 `kv_task_engine` 可能没有被可靠关闭。
- 新逻辑改为统一 `self._cleanup()`，确保：
  - `kv_task_engine.shutdown()` 只执行一次（幂等保护）
  - 关闭 `client_manager` 持有的 ZMQ socket
  - 关闭 `recv_from_client` socket
  - `context.term()`
- `__del__` 也改为调用同一个 `_cleanup()`，避免遗漏释放。

这项修复直接针对“多轮后资源累积”问题。

---

## 当前现象与结论

- 单次 smoke 运行可成功。
- 多轮长跑（尤其在资源紧张环境）仍有概率在后续轮次触发 `cudaHostRegister failed with error code 1`。
- 这更接近环境/资源上限问题，不完全是 benchmark 逻辑问题。

---

## 建议运行方式（当前）

推荐先用 smoke 配置验证稳定性：

```bash
PYTHONPATH=/workspace/FlexKV python3 benchmarks/kvmanager_mode_compare/run_compare.py \
  --modes direct,server_client \
  --config benchmarks/kvmanager_mode_compare/smoke_config.yml \
  --warmup 2 \
  --repeats 10 \
  -o /workspace/mode_compare_result_multi.jsonl \
  --summary-output /workspace/mode_compare_result_multi_summary.json
```

若仍出现后段轮次失败，可采用“每轮前清理残留进程”的方式降低累积影响。

---

## 可选回退说明

如果后续决定不保留 FlexKV 内核改动（只保留 benchmark 侧），可回退：

- `flexkv/transfer/worker.py`

再单独保留：

- `benchmarks/kvmanager_mode_compare/run_compare.py`
- `benchmarks/kvmanager_mode_compare/formal_lite_config.yml`


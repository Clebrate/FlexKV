"""
kvmanager_mode_compare — 测量范围（与 direct / server_client 对比相关的耗时）

目标
----
在**不依赖 vLLM 等推理框架**的前提下，用同一套 workload 对比 FlexKV 两种路径的耗时差异：

- **direct**：进程内 KVManager 与 GPU 侧协同为主（FLEXKV_SERVER_CLIENT_MODE=0，且单实例等条件下
  server_client_mode 为 false，见 flexkv.kvmanager.KVManager）。
- **server_client**：经 FlexKV 的 server/client 与 TP 子进程注册 GPU block、走 ZMQ 等路径
  （FLEXKV_SERVER_CLIENT_MODE=1，并配置 FLEXKV_SERVER_RECV_PORT 等）。

注意：flexkv 在首次 import 时从环境读取 GLOBAL_CONFIG_FROM_ENV；**切换模式应在全新 Python
进程中进行**（子进程或两次独立命令），同一进程内改环境变量不可靠。

对比的「阶段」拆解
--------------------
下列项与现有 benchmark_single_batch 流程对齐，可逐项计时（perf_counter）或拆成多次实验：

1. **进程与连接就绪**
   - KVManager 构造 + start() 到 is_ready() 为 True 的 wall time（含 server 模式下的监听就绪）。
   - **TP client**：子进程启动 + register_to_server 完成（server_client 下差异通常最大）。

2. **Put 路径**（先填充缓存 / 模拟写入）
   - put_async（逐条或批量）提交阶段耗时。
   - wait(..., completely=True) 直到完成：含调度、CPU/GPU/SSD 传输等（direct vs client 的 IPC/ZMQ 开销叠在这里）。

3. **Get 路径**（可选 clear_cpu_cache 以改变命中层级）
   - get_match：仅前缀匹配 / 构图阶段耗时（若 API 允许与后续 launch 分开统计）。
   - launch + wait：真正把 KV 搬到 GPU slot 的端到尾耗时。

4. **可选扩展**（需要代码里加埋点或多次实验才能干净拆开时再补）
   - 纯「消息往返」：最小 no-op 或仅注册类请求（若仓库后续暴露专用接口）。
   - 与 transfer worker 相关的带宽：可对照 benchmarks/benchmark_workers.py，与本目录的「模式差异」互补。

控制变量（公平对比）
--------------------
- 同一 `ModelConfig` / `CacheConfig`（同一 YAML），同一 batch_size、sequence_length、
  tokens_per_block、cache_ratio、dtype、tp_size。
- 同一 GPU 与驱动环境；server_client 需与文档一致配置 FLEXKV_SERVER_RECV_PORT。
- 预热：每种模式先跑 1～N 轮 discard，再计分。
- 报告：除总耗时外，建议记录 put/get 分段、token 数、有效带宽（GB/s），便于排除数据量不同带来的假象。

输出约定（实现 run_compare.py 时）
---------------------------------
- JSON 或 JSONL：每行包含 mode、阶段名、毫秒、batch/seq、配置摘要、可选重复序号。
- 或打印表格后可选 --output result.jsonl。

下一步
------
在本目录实现 run_compare.py：对每种 mode 用 subprocess 起独立子进程跑同一 workload，聚合结果。
可参考 benchmarks/vllm_server_client_mode/benchmark_single_batch.py 的 workload 形状，但显式设置
FLEXKV_SERVER_CLIENT_MODE 与端口，并拆分计时点。
"""

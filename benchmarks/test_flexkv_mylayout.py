import tempfile
from multiprocessing import Process
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch

from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import flexkv_logger
from flexkv.common.config import ModelConfig, CacheConfig
from utils import load_config
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus

flexkv_logger.set_level("INFO")

@dataclass
class MyKVCacheLayout(KVCacheLayout):
    def __post_init__(self) -> None:
        self._kv_shape = torch.Size([self.num_layer,
                                     self.num_block,
                                     self._kv_dim,
                                     self.tokens_per_block,
                                     self.num_head,
                                     self.head_size])

    def get_layer_stride(self) -> int:
        print(f">>>[DEV] MyKVCacheLayout.get_layer_stride called <<<")
        return self.kv_shape[1:].numel()

    def get_block_stride(self) -> int:
        print(f">>>[DEV] MyKVCacheLayout.get_block_stride called <<<")
        return self.kv_shape[2:].numel()

    def get_kv_stride(self) -> int:
        print(f">>>[DEV] MyKVCacheLayout.get_block_stride called <<<")
        return self.kv_shape[3:].numel()

@dataclass
class BenchmarkConfig:
    num_layers_to_transfer: int
    batch_size: int
    sequence_length: int
    sequence_length_step: int
    slot_mapping_seed: int
    cache_ratio: float
    clear_cpu_cache: bool

def run_tp_client(dp_client_id, tp_rank, gpu_register_port, model_config, cache_config, gpu_kv_layout, gpu_blocks_for_tp):
    """Run tp_client process"""
    device_id = tp_rank + dp_client_id * model_config.tp_size
    tp_client = KVTPClient(gpu_register_port, dp_client_id, device_id)

    time.sleep(10)
    tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)
    time.sleep(10)


def build_aligned_sequence_lengths(
    batch_size: int,
    min_sequence_length: int,
    sequence_length_step: int,
    tokens_per_block: int,
) -> List[int]:
    if min_sequence_length % tokens_per_block != 0:
        raise ValueError(
            f"min_sequence_length must be {tokens_per_block}-aligned, got {min_sequence_length}"
        )
    if sequence_length_step % tokens_per_block != 0:
        raise ValueError(
            f"sequence_length_step must be {tokens_per_block}-aligned, got {sequence_length_step}"
        )
    return [min_sequence_length + i * sequence_length_step for i in range(batch_size)]


# 不同位置的回填 + noncontiguous page_id
def build_disjoint_shuffled_slot_mappings(
    sequence_lengths: List[int],
    total_slots: int,
    tokens_per_block: int,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if total_slots % tokens_per_block != 0:
        raise ValueError(
            f"total_slots must be divisible by tokens_per_block: "
            f"total_slots={total_slots}, tokens_per_block={tokens_per_block}"
        )
    for seq_len in sequence_lengths:
        if seq_len % tokens_per_block != 0:
            raise ValueError(
                f"sequence length must be divisible by tokens_per_block: "
                f"seq_len={seq_len}, tokens_per_block={tokens_per_block}"
            )

    total_tokens = sum(sequence_lengths)
    total_blocks = total_slots // tokens_per_block
    total_blocks_needed = total_tokens // tokens_per_block
    if total_blocks < 2 * total_blocks_needed:
        raise ValueError(
            f"not enough blocks for disjoint put/get mappings: "
            f"total_blocks={total_blocks}, required={2 * total_blocks_needed}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    all_blocks = torch.randperm(total_blocks, generator=generator, dtype=torch.int64)
    put_pool_blocks = all_blocks[:total_blocks_needed]
    get_pool_blocks = all_blocks[total_blocks_needed : 2 * total_blocks_needed]
    token_offsets = torch.arange(tokens_per_block, dtype=torch.int64)

    def split_and_shuffle(pool_blocks: torch.Tensor) -> List[torch.Tensor]:
        mappings = []
        cursor = 0
        for seq_len in sequence_lengths:
            seq_blocks = seq_len // tokens_per_block
            chosen_blocks = pool_blocks[cursor : cursor + seq_blocks]
            cursor += seq_blocks
            # Keep token offsets ordered within each block since current FlexKV mapping
            # is block-granular (slot_mapping_to_block_ids uses slot_mapping[::tokens_per_block]).
            block_perm = torch.randperm(seq_blocks, generator=generator)
            chosen_blocks = chosen_blocks[block_perm]
            seq_slots = (
                chosen_blocks[:, None] * tokens_per_block + token_offsets[None, :]
            ).reshape(-1)
            mappings.append(seq_slots.clone())
        return mappings

    return split_and_shuffle(put_pool_blocks), split_and_shuffle(get_pool_blocks)

#对比回填
def copy_slot_value(
    src_layer_tensor: torch.Tensor,
    dst_layer_tensor: torch.Tensor,
    src_slot: int,
    dst_slot: int,
    tokens_per_block: int,
) -> None:
    src_block, src_token = divmod(int(src_slot), tokens_per_block)
    dst_block, dst_token = divmod(int(dst_slot), tokens_per_block)
    dst_layer_tensor[dst_block, :, dst_token, :, :] = src_layer_tensor[src_block, :, src_token, :, :]


def benchmark_flexkv(model_config: ModelConfig,
                     cache_config: CacheConfig,
                     benchmark_config: BenchmarkConfig,
                     ):
    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError(f"tp_size {model_config.tp_size} * dp_size {model_config.dp_size} is greater than "
                         f"the number of available GPUs {torch.cuda.device_count()}")
    print(f"{benchmark_config = }")
    kvmanager = KVManager(model_config, cache_config)
    kvmanager.start()

    tp_client_processes = []

    batch_size = benchmark_config.batch_size
    sequence_lengths = build_aligned_sequence_lengths(
        batch_size=batch_size,
        min_sequence_length=benchmark_config.sequence_length,
        sequence_length_step=benchmark_config.sequence_length_step,
        tokens_per_block=cache_config.tokens_per_block,
    )
    print("====+++++====")
    print(f"8 sequence lengths: {sequence_lengths}")
    print(batch_size)
    print(cache_config.tokens_per_block)
    print("====+++++====")
    base_required_gpu_blocks = (
        sum(sequence_lengths) + cache_config.tokens_per_block - 1
    ) // cache_config.tokens_per_block
    num_required_gpu_blocks = max(
        int(base_required_gpu_blocks * 1.5),
        base_required_gpu_blocks + 64,
    )
    cache_config.num_gpu_blocks = num_required_gpu_blocks
    print(
        f"allocate {num_required_gpu_blocks} gpu blocks for benchmark "
        f"(base_required={base_required_gpu_blocks})"
    )
    total_slots = cache_config.num_gpu_blocks * cache_config.tokens_per_block
    put_slot_mappings, get_slot_mappings = build_disjoint_shuffled_slot_mappings(
        sequence_lengths=sequence_lengths,
        total_slots=total_slots,
        tokens_per_block=cache_config.tokens_per_block,
        seed=benchmark_config.slot_mapping_seed,
    )
    for i in range(batch_size):
        put_set = set(put_slot_mappings[i].tolist())
        get_set = set(get_slot_mappings[i].tolist())
        if put_set.intersection(get_set):
            raise AssertionError(f"put/get slot mapping overlap detected for sample {i}")

    # Create GPU blocks for this tp_rank in the tp_client process
    gpu_kv_layout = MyKVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    print(gpu_kv_layout.kv_shape)
    num_gpu_blocks = cache_config.num_gpu_blocks
    gpu_kv_layout._kv_shape = torch.Size([
        model_config.num_layers, 
        cache_config.num_gpu_blocks, 
        gpu_kv_layout._kv_dim,
        cache_config.tokens_per_block,
        model_config.num_kv_heads,
        model_config.head_size,
    ])
    print(gpu_kv_layout.kv_shape)

    gpu_blocks_for_tp = []
    for _ in range(model_config.num_layers):
        gpu_blocks_for_tp.append(
            torch.randn(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(0)
        )
    backups = [ t.clone() for t in gpu_blocks_for_tp ]
    for i in range(model_config.num_layers):
        assert torch.allclose(backups[i], gpu_blocks_for_tp[i])
    print("All backuped")
    
    run_tp_client(0, 0, kvmanager.gpu_register_port, model_config, cache_config, gpu_kv_layout, gpu_blocks_for_tp)

    while not kvmanager.is_ready():
        time.sleep(3)
        flexkv_logger.info("waiting for flexkv to be ready")
    flexkv_logger.info("flexkv is ready")

    batch_sequence_tensor = []
    batch_slot_mapping = []

    # generate requests
    for i in range(batch_size):
        seq_len = sequence_lengths[i]
        batch_sequence_tensor.append(torch.arange(i * 100000, i * 100000 + seq_len, dtype=torch.int64))
        batch_slot_mapping.append(put_slot_mappings[i])
        print(batch_slot_mapping[-1])
        print(f"get_slot_mapping[{i}] = {get_slot_mappings[i]}")

    # benchmark put
    start_time = time.time()
    batch_put_ids = []
    if benchmark_config.cache_ratio > 0:
        for i in range(batch_size):
            cache_length = int(sequence_lengths[i] * benchmark_config.cache_ratio)
            task_id = kvmanager.put_async(batch_sequence_tensor[i][:cache_length],
                                          batch_slot_mapping[i][:cache_length],
                                          token_mask=None)
            batch_put_ids.append(task_id)
    put_result = kvmanager.wait(batch_put_ids, completely=True)
    end_time = time.time()

    if benchmark_config.clear_cpu_cache:
        kvmanager._clear_cpu_cache()

    elapsed_time_put = end_time - start_time
    put_tokens = 0
    for _, response in put_result.items():
        if response.status == KVResponseStatus.SUCCESS:
            put_tokens += response.return_mask.sum().item()
    print(put_tokens)

    for i in range(model_config.num_layers):
        gpu_blocks_for_tp[i].copy_(torch.zeros_like(gpu_blocks_for_tp[i]))

    for i in range(model_config.num_layers):
        assert not torch.allclose(backups[i], gpu_blocks_for_tp[i])
    print("All cleared")

    all_tokens = 0
    start_time = time.time()
    batch_get_ids = []
    for i in range(batch_size):
        all_tokens += len(batch_sequence_tensor[i])
        task_id, _ = kvmanager.get_match(batch_sequence_tensor[i],
                                      token_mask=None)
        batch_get_ids.append(task_id)
    get_match_time = time.time() - start_time
    kvmanager.launch(batch_get_ids, get_slot_mappings)
    get_result = kvmanager.wait(batch_get_ids)
    print(get_result, type(get_result))
    elapsed_time_get = time.time() - start_time
    cached_tokens = 0
    for _, response in get_result.items():
        if response.status == KVResponseStatus.SUCCESS:
            cached_tokens += response.return_mask.sum().item()
    print(cached_tokens)

    expected_blocks = [torch.zeros_like(layer_tensor) for layer_tensor in backups]
    for i, task_id in enumerate(batch_get_ids):
        response = get_result[task_id]
        if response.status != KVResponseStatus.SUCCESS:
            raise RuntimeError(f"get task failed: task_id={task_id}, status={response.status}")
        return_mask = response.return_mask
        if isinstance(return_mask, torch.Tensor):
            return_mask_cpu = return_mask.detach().to("cpu")
        else:
            return_mask_cpu = torch.as_tensor(return_mask)
        hit_positions = torch.nonzero(return_mask_cpu, as_tuple=False).squeeze(-1).tolist()
        for pos in hit_positions:
            src_slot = int(put_slot_mappings[i][pos].item())
            dst_slot = int(get_slot_mappings[i][pos].item())
            for layer_idx in range(model_config.num_layers):
                copy_slot_value(
                    src_layer_tensor=backups[layer_idx],
                    dst_layer_tensor=expected_blocks[layer_idx],
                    src_slot=src_slot,
                    dst_slot=dst_slot,
                    tokens_per_block=cache_config.tokens_per_block,
                )

    for i in range(model_config.num_layers):
        assert torch.allclose(expected_blocks[i], gpu_blocks_for_tp[i])
        print(torch.max(torch.abs(expected_blocks[i] - gpu_blocks_for_tp[i])))
    print("All matched")

    # shutdown_tp_client(tp_client_processes)
    kvmanager.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmarks/example_config.yml")
    # benchmark config
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--sequence-length-step", type=int, default=32)
    parser.add_argument("--slot-mapping-seed", type=int, default=2026)
    parser.add_argument("--cache-ratio", type=float, default=1)
    parser.add_argument("--clear-cpu-cache", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    benchmark_config = BenchmarkConfig(
        num_layers_to_transfer=args.num_layers,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        sequence_length_step=args.sequence_length_step,
        slot_mapping_seed=args.slot_mapping_seed,
        cache_ratio=args.cache_ratio,
        clear_cpu_cache=args.clear_cpu_cache
    )
    model_config, cache_config = load_config(args.config)
    #cache_config.num_cpu_blocks = 8192 - 2048
    # pad sequence length to divisible by tokens_per_block
    benchmark_config.sequence_length = \
        ((benchmark_config.sequence_length - 1) // cache_config.tokens_per_block + 1) * cache_config.tokens_per_block

    benchmark_flexkv(model_config, cache_config, benchmark_config)
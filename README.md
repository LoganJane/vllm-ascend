<!-- markdownlint-disable MD001 MD041 -->
<h1 align="center">
Kimi-K2.5 模型基于vLLM-Ascend 推理指导
</h1>

## 一、准备运行环境

**表 1**  版本配套表

| 配套  | 版本 | 环境准备指导 |
| ----- | ----- |-----|
| Python | 3.11.10 | - |
| torch | 2.9.0 | - |
| transformers | 4.57.6 | - |

### 1.1 获取vllm-ascend镜像

- [镜像链接](https://quay.io/repository/ascend/vllm-ascend?tab=tags)：https://quay.io/repository/ascend/vllm-ascend?tab=tags
- 下载镜像命令
```shell
docker pull quay.io/ascend/vllm-ascend:v0.14.0rc1
```

### 1.2 特性分支

- vllm：https://github.com/LoganJane/vllm/tree/main
- vllm-ascend：https://github.com/LoganJane/vllm-ascend/tree/main

### 1.3 社区适配PR

- vllm：https://github.com/vllm-project/vllm/pull/34501
- vllm-ascend：https://github.com/vllm-project/vllm-ascend/pull/6755

### 1.4 安装 vllm & vllm-ascend

```shell
# 卸载镜像中vllm/vllm-ascend
pip uninstall -y vllm vllm_ascend
```

```shell
# 安装vllm
git clone https://github.com/LoganJane/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
```

```shell
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# 安装vllm-ascend
git clone https://github.com/LoganJane/vllm-ascend.git
cd vllm-ascend
pip install -v -e .
```

## 二、下载权重

### 2.1 vLLM-Ascend w4a8权重

- ModelScope

|  模型 | 链接  |
| ------------ | ------------ |
| Eco-Tech/Kimi-K2.5-W4A8  |  [ModelScope](https://modelscope.cn/models/Eco-Tech/Kimi-K2.5-W4A8/files) |

### 2.2 Kimi-K2.5 官方权重

- Huggingface

|  模型 | 链接  |
| ------------ | ------------ |
| moonshotai/Kimi-K2.5  |  [🤗huggingface](https://huggingface.co/moonshotai/Kimi-K2.5/tree/main) |

- ModelScope

|  模型 | 链接  |
| ------------ | ------------ |
| moonshotai/Kimi-K2.5  |  [ModelScope](https://modelscope.cn/models/moonshotai/Kimi-K2.5/files) |

## 三、Atlas 800I A3 单机混部部署

### 3.1 w4a8量化权重启动命令

```shell
#!/bin/sh
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_TORCH_PROFILER_WITH_STACK=0

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_BALANCE_SCHEDULING=1
export HCCL_BUFFSIZE=1536
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /weights/Kimi-K2.5-W4A8 \
    --served-model-name kimi \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --quantization ascend \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --port 8008 \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --max-num-batched-tokens 12288 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path / \
    --seed 42 \
    --async-scheduling \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data \
    --compilation-config '{"cudagraph_capture_sizes":[256,192,160,128,96,64,32,16,8,4,2,1], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false}}'
```

### 3.2 官方原始权重启动命令

```shell
#!/bin/sh
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_TORCH_PROFILER_WITH_STACK=0

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve /weights/Kimi-K2.5 \
    --served-model-name kimi \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --port 8008 \
    --max-num-seqs 256 \
    --max-model-len 32768 \
    --max-num-batched-tokens 12288 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path / \
    --seed 42 \
    --async-scheduling \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data \
    --compilation-config '{"cudagraph_capture_sizes":[256,192,160,128,96,64,32,16,8,4,2,1], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false}}'
```

### 3.3 发送多模态请求执行推理

```shell
curl http://localhost:8008/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "file:///datasets/test.jpg",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": "请描述图片中的内容。"
            }]
        }],
        "max_tokens": 1024
    }'
```

## 四、Atlas 800I A3 双机PD分离部署

### 4.1 PD proxy

```shell
python vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port 8008 \
    --prefiller-hosts 102.34.56.78 \
    --prefiller-port 1025 \
    --decoder-hosts 102.34.56.79 \
    --decoder-ports 1026
```

### 4.2 Prefill node 0

```shell
nic_name="enp48s3u1u1"
local_ip="102.34.56.78"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000

export VLLM_WORKER_MULTIPROC_METHOD="fork"
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /weights/Kimi-K2.5-W4A8 \
    --host 0.0.0.0 \
    --port 1025 \
    --quantization ascend \
    --served-model-name kimi \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path / \
    --seed 42 \
    --async-scheduling \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false}}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "30100",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                },
                "decode": {
                        "dp_size": 4,
                        "tp_size": 4
                }
        }
    }'
```

### 4.3 Decode node 0

```shell
nic_name="enp48s3u1u1"
local_ip="102.34.56.79"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000

export VLLM_WORKER_MULTIPROC_METHOD="fork"
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /weights/Kimi-K2.5-W4A8 \
    --host 0.0.0.0 \
    --port 1026 \
    --quantization ascend \
    --served-model-name kimi \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --max-num-batched-tokens 128 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path / \
    --seed 42 \
    --async-scheduling \
    --mm-processor-cache-type shm \
    --mm-encoder-tp-mode data \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false}}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30200",
    "engine_id": "1",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                },
                "decode": {
                        "dp_size": 4,
                        "tp_size": 4
                }
        }
    }'
```

### 4.4 发送多模态请求执行推理

```shell
curl http://localhost:8008/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "file:///datasets/test.jpg",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": "请描述图片中的内容。"
            }]
        }],
        "max_tokens": 1024
    }'
```

# Qwen3-VL Usage Guide
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) is the most powerful vision-language model in the Qwen series to date created by Alibaba Cloud. 

This generation delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

Available in Dense and MoE architectures that scale from edge to cloud, with Instruct and reasoning‑enhanced Thinking editions for flexible, on‑demand deployment.


## Installing vLLM

```bash
uv venv
source .venv/bin/activate

# Install vLLM >=0.11.0
uv pip install -U vllm

# Install Qwen-VL utility library (recommended for offline inference)
uv pip install qwen-vl-utils==0.0.14
```


## Running Qwen3-VL


### Qwen3-VL-235B-A22B-Instruct
This is the Qwen3-VL flagship MoE model, which requires a minimum of 8 GPUs, each with at least 80 GB of memory (e.g., A100, H100, or H200). On some types of hardware the model may not launch successfully with its default setting. Recommended approaches by hardware type are:

- **H100 with `fp8`**: Use FP8 checkpoint for optimal memory efficiency.
- **A100 & H100 with `bfloat16`**: Either reduce `--max-model-len` or restrict inference to images only.
- **H200 & B200**: Run the model out of the box, supporting full context length and concurrent image and video processing.

See sections below for detailed launch arguments for each configuration. We are actively working on optimizations and the recommended ways to launch the model will be updated accordingly.

<details>
<summary>H100 (Image + Video Inference, FP8)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling
```

</details>


<details>
<summary>H100 (Image Inference, FP8, TP4)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128
```

</details>


<details>
<summary>A100 & H100 (Image Inference, BF16)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling
```

</details>


<details>
<summary>A100 & H100 (Image + Video Inference, BF16)</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 128000 \
  --async-scheduling
```

</details>


<details>
<summary>H200 & B200</summary>

```bash
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --async-scheduling
```

</details>

> ℹ️ **Note**  
> Qwen3-VL-235B-A22B-Instruct also excels on text-only tasks, ranking as the [#1 open model on text by lmarena.ai](https://x.com/arena/status/1973151703563460942) at the time this guide was created.  
> You can enable text-only mode by passing `--limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 0`, which skips the vision encoder and multimodal profiling to free up memory for additional KV cache.


### Configuration Tips
- It's highly recommended to specify `--limit-mm-per-prompt.video 0` if your inference server will only process image inputs since enabling video inputs consumes more memory reserved for long video embeddings. Alternatively, you can skip memory profiling for multimodal inputs by `--skip-mm-profiling` and lower `--gpu-memory-utilization` accordingly at your own risk.
- To avoid undesirable CPU contention, it's recommended to limit the number of threads allocated to preprocessing by setting the environment variable `OMP_NUM_THREADS=1`. This is particulaly useful and shows significant throughput improvement when deploying multiple vLLM instances on the same host.
- You can set `--max-model-len` to preserve memory. By default the model's context length is 262K, but `--max-model-len 128000` is good for most scenarios.
- Specifying `--async-scheduling` improves the overall system performance by overlapping scheduling overhead with the decoding process. **Note: With vLLM >= 0.11.1, compatibility has been improved for structured output and sampling with penalties, but it may still be incompatible with speculative decoding (features merged but not yet released).** Check the latest releases for continued improvements.
- Specifying `--mm-encoder-tp-mode data` deploys the vision encoder in a data-parallel fashion for better performance. This is because the vision encoder is very small, thus tensor parallelism brings little gain but incurs significant communication overhead. Enabling this feature does consume additional memory and may require adjustment on `--gpu-memory-utilization`.
- If your workload involves mostly **unique** multimodal inputs only, it is recommended to pass `--mm-processor-cache-gb 0` to avoid caching overhead. Otherwise, specifying `--mm-processor-cache-type shm` enables this experimental feature which utilizes host shared memory to cache preprocessed input images and/or videos which shows better performance at a high TP setting.
- vLLM supports Expert Parallelism (EP) via `--enable-expert-parallel`, which allows experts in MoE models to be deployed on separate GPUs for better throughput. Check out [Expert Parallelism Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html) for more details.
- You can use [benchmark_moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py) to perform MoE Triton kernel tuning for your hardware.
- You can further extend the model's context window with `YaRN` by passing `--rope-scaling '{"rope_type":"yarn","factor":3.0,"original_max_position_embeddings": 262144,"mrope_section":[24,20,20],"mrope_interleaved": true}' --max-model-len 1000000`


### Benchmark on VisionArena-Chat Dataset

Once the server for the `Qwen3-VL-235B-A22B-Instruct` model is running, open another terminal and run the benchmark client:

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --request-rate 20
```

### Consume the OpenAI API Compatible Server
```python
import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Read all the text in the image."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

For more usage examples, check out the [vLLM user guide for multimodal models](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) and the [official Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)!



## AMD GPU Support

These steps are for **MI300X / MI325X / MI355X** using the **vLLM ROCm Docker image**. Two Hugging Face checkpoints are covered:

| Variant | Hugging Face model id | Notes |
|--------|------------------------|--------|
| **BF16 ** | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Default MoE instruct weights in BF16 |
| **FP8** | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | Same architecture; FP8 weights for optimal memory efficiency |

**Workflow:** complete **Step 1** (container shell). Run **exactly one** of the **Step 2** launch blocks (BF16 **or** FP8). In **Step 3**, set `--model` on `vllm bench serve` to the **same** id you used in `vllm serve`.

### Step 1: Install the vLLM ROCm Docker image

Use the official image [`vllm/vllm-openai-rocm` on Docker Hub](https://hub.docker.com/r/vllm/vllm-openai-rocm). From the host, start an interactive shell in the container. **Steps 2 and 3** assume commands run **inside** that shell unless stated otherwise (for example, the benchmark client may run on the host or another machine that can reach the API).

To build or customize images from source, see the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#pre-built-images).

To access private Hugging Face assets, export `HF_TOKEN` on the host before `docker run`. If you do not need a token, remove the `--env` line.

```bash
docker run -it --rm \
  --name vllm-openai-rocm \
  --entrypoint bash \
  --device /dev/dri:/dev/dri \
  --device /dev/kfd:/dev/kfd \
  --group-add video \
  --ipc host \
  --network host \
  --security-opt apparmor=unconfined \
  --security-opt seccomp=unconfined \
  --shm-size 128G \
  -v /data/huggingface/hub:/root/.cache/huggingface/hub \
  --env "HF_TOKEN=$HF_TOKEN" \
  vllm/vllm-openai-rocm:v0.22.0
```

Replace the image tag (`v0.22.0`) if you use a different release (for example `v0.23.0`), and adjust `--name`, the Hugging Face cache mount (`-v`), and other flags to match your environment.

### Step 2: Start the vLLM server

Run **one** of the following launch blocks **inside the container** from Step 1. Environment variables apply to the shell session; use `export` so they are visible to `vllm serve`.

#### BF16 — `Qwen/Qwen3-VL-235B-A22B-Instruct`

Serve the **BF16** checkpoint. Change `--max-model-len`, `--limit-mm-per-prompt`, and batch settings if you need more headroom or multimodal video.

```bash
export SAFETENSORS_FAST_GPU="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MHA="1"
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT="1"
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling \
  --gpu-memory-utilization 0.94 \
  --max-model-len 32768 \
  --max-num-seqs 10240 \
  --max-num-batched-tokens 32768 \
  --attention-backend ROCM_AITER_FA 
```

#### FP8 — `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`

Serve the **FP8** checkpoint. This example uses chunked prefill, FP8 KV cache, and ROCm-oriented compilation and attention settings.

```bash
export SAFETENSORS_FAST_GPU="1"
export HIP_FORCE_DEV_KERNARG="1"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ROCM_USE_AITER="1"
export VLLM_ROCM_USE_AITER_MHA="1"
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT="1"
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --async-scheduling \
  --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization 0.94 \
  --max-model-len 32768 \
  --max-num-seqs 10240 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype fp8 \
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["+rms_norm", "+quant_fp8"]}' \
  --attention-backend ROCM_AITER_FA 
```

**Tuning (FP8 launch above):** If time-per-output-token (TPOT) is too high, reduce `--max-num-batched-tokens`. For longer contexts, retune `--max-num-seqs` and `--max-num-batched-tokens`. Available options for `--attention-backend` are: `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN` & `TRITON_ATTN`. For **image-only** serving, narrow multimodal limits (for example set video to `0` in `--limit-mm-per-prompt`) to save memory.

When startup finishes, the server log should show `INFO: Application startup complete.`


### Step 3: Run benchmarks

After the server is accepting traffic, run benchmarks from a **separate** terminal. Set `--model` to the **same** Hugging Face id you used in **Step 2** (`Qwen/Qwen3-VL-235B-A22B-Instruct` or `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`).

The examples use `vllm bench serve` with the **`random-mm`** synthetic multimodal workload ([`RandomMultiModalDataset`](https://docs.vllm.ai/en/latest/api/vllm/benchmarks/datasets/#vllm.benchmarks.datasets.RandomMultiModalDataset)).

#### BF16 server — `--model Qwen/Qwen3-VL-235B-A22B-Instruct`

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --num-prompts 1000 \
  --num-warmups 10 \
  --request-rate 20 \
  --dataset-name random-mm \
  --random-input-len 1024 \
  --random-output-len 512 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(512, 512, 1): 1.0}' \
  --ignore-eos
```

#### FP8 server — `--model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`

```bash
vllm bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --num-prompts 1000 \
  --num-warmups 10 \
  --request-rate 20 \
  --dataset-name random-mm \
  --random-input-len 1024 \
  --random-output-len 512 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(512, 512, 1): 1.0}' \
  --ignore-eos
```
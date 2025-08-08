# llama-server 使用指南

## 简介

llama-server 是一个高性能推理服务端，针对昇腾环境进行了特殊优化，提供与 OpenAI API 兼容的接口。服务器支持多种大语言模型，如 DeepSeek-V2-Lite 等，并提供高效的文本生成和聊天功能。

环境在 Kunpeng 920 + Ascend910B3 (CANN 8.2.RC1.alpha001) 上已得到验证，将整个项目按照如下命令编译即可：
```bash
cmake -B build -DGGML_CUDA=OFF -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2" -DCMAKE_C_FLAGS="-O2"
cmake --build build -j
```

注意需要预先安装好 CANN，G++ 以及 mpicxx 等依赖。（注意，我们暂时不支持毕昇编译器，因为目前仍不支持mpi）

## 配置文件

服务器使用 YAML 格式的配置文件来设置模型和服务器参数。示例配置文件位于：

```bash
configs/config_deepseek_v2_lite.yaml
```

具体运行方式为

```bash
ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-server --config /path/to/configs/config_deepseek_v2_lite.yaml
```

对于单卡放不下的模型，我们首先支持最简单的按层级切分，会默认使用环境变量中所有可用的卡进行切分，以 Deepseek V2 为例，运行的方式如下：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./build/bin/llama-server --config /path/to/configs/config_deepseek_v2.yaml
```

### EP + TP

目前只支持比较简单的方法定制专家并行与张量并行，后续会支持更多自定义方法。

具体运行方式为（ `np` 中指定使用的 NPU 个数），例子如下：

```bash
mpirun -np 2 ./build/bin/llama-mpi-server --config /path/to/configs/config_deepseek_v2_lite_ep.yaml
# 在两张卡上运行 EP + TP 的 Deepseek V2 Lite 方式
```

### 主要配置参数

### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `main_gpu` | int | 主 GPU 设备编号 |
| `n_gpu_layers` | int | 在 GPU 上加载的模型层数 |
| `split_mode` | string | 模型在多个 GPU 上的分割方式 |
| `tensor_split` | array | 张量在 GPU 上的分布 |
| `use_mmap` | bool | 使用内存映射加速模型加载 |
| `use_mlock` | bool | 将模型锁定在内存中 |
| `check_tensors` | bool | 验证张量数据 |
| `n_threads` | int | 计算使用的线程数 |
| `n_threads_batch` | int | 批处理使用的线程数 |
| `defrag_thold` | float | 内存碎片整理阈值 |
| `no_perf` | bool | 禁用性能指标 |
| `enable_mla` | bool | 为 DeepSeek 启用 MLA 加速 |
| `enable_fused_moe` | bool | 使用融合 MoE 加速 |
| `offload_input` | bool | 将输入卸载到 GPU |
| `enable_ge` | bool | 为 DeepSeek 启用 GE 加速 |
| `display_chat` | bool | 在控制台显示聊天信息 |
| `presample_count` | int | NPU 上的预采样数量，-1 表示禁用 |
| `n_ctx` | int | 上下文窗口大小 |
| `n_batch` | int | 提示处理的逻辑批量大小 |
| `n_parallel` | int | 并行解码的序列数量 |
| `model` | string | 模型文件路径 |
| `model_alias` | string | 模型别名 |
| `special` | bool | 启用特殊 token 输出 |
| `cont_batching` | bool | 动态插入新序列进行解码 |
| `reranking` | bool | 在服务器上启用重排序支持 |
| `port` | int | 服务器监听的网络端口 |
| `n_threads_http` | int | 处理 HTTP 请求的线程数 |
| `hostname` | string | 服务器绑定的主机名 |
| `chat_template` | string | 聊天模板 |

## API 接口

服务器提供与 OpenAI API v1 兼容的接口，支持以下接口：

### 聊天接口

- **端点**: `/v1/chat/completions`
- **方法**: POST
- **请求格式**:

```json
{
  "model": "DeepSeek-V2-Lite",
  "messages": [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好，请介绍一下自己！"}
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": true
}
```

## 参数说明

- `model`: 使用的模型名称
- `messages`/`prompt`: 输入的消息或提示文本
- `temperature`: 采样温度，控制输出的随机性（0-1）
- `top_p`: 核采样参数，控制输出的多样性（0-1）
- `max_tokens`/`n_predict`: 生成的最大 token 数量
- `stream`: 是否使用流式输出（实时返回生成结果）

## 示例代码

### Python 客户端示例

```python
import requests
import json
import sseclient

# 聊天完成（流式）
def chat_completion_stream():
    url = "http://127.0.0.1:8080/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "DeepSeek-V2-Lite",
        "messages": [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": "请简要介绍一下中国的历史。"}
        ],
        "temperature": 0.7,
        "stream": True
    }
    
    response = requests.post(url, headers=headers, json=data, stream=True)
    client = sseclient.SSEClient(response)
    
    for event in client.events():
        if event.data != "[DONE]":
            chunk = json.loads(event.data)
            content = chunk["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
        else:
            print("\n[完成]")

# 文本补全（非流式）
def text_completion():
    url = "http://127.0.0.1:8080/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "DeepSeek-V2-Lite",
        "prompt": "人工智能的未来发展趋势是",
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False
    }
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    print(result["choices"][0]["text"])

# 运行示例
if __name__ == "__main__":
    print("流式聊天示例：")
    chat_completion_stream()
    
    print("\n文本补全示例：")
    text_completion()
```

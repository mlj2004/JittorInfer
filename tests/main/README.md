# tests/main

对于需要需要调试模型或者验证模型可用性的情况，我们提供了简单的测试脚本。这个部分最小实现了简单的对话功能。

如果需要使用这个模块的代码，需要开启LLAMA_BUILD_TESTS选项。

## 单并发测试

单并发测试的入口是`tests/main/single_sequence.cpp`。各项参数硬编码于这个文件当中，包括模型路径、并发数等超参数。

这个脚本会与模型进行一段简单的对话。这个测试的可执行文件位于：
```
build/bin/llama-test-single-sequence
```

## 多并发测试

多并发测试的入口是`tests/main/multi_sequence.cpp`。各项参数硬编码于这个文件当中，包括模型路径、并发数等超参数。

这个脚本提供了多条输入语句，模型会并行处理这些语句。这个测试的可执行文件位于：
```
build/bin/llama-test-multi-sequence
```


# 参考资料 - 性能优化

## 核心技术
- KV Cache优化: FlashAttention-3 (2024)
- 模型量化: AWQ, GPTQ, bitsandbytes 4-bit
- 批处理: Continuous Batching (vLLM)

## 缓存策略
- GPTCache: [github.com/zilliztech/GPTCache](https://github.com/zilliztech/GPTCache)
- Semantic Cache with Redis + Embedding

## 2026优化框架
- FlashMLA (DeepSeek): 35x推理加速
- vLLM Prefix Caching: 自动复用KV Cache

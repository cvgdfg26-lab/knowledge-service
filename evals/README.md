先上传、解析、审核、发布样例资料，再将真实 version_id 与原文定位填入 JSONL。
示例仅展示格式，不计作完整评测集。按需求文档另外准备至少 150 道人工标注题，并按来源文档划分 dev/test，避免同一来源泄漏。

四组配置分别设 `mode=dense/sparse/hybrid/hybrid`，前三组 `rerank=false`，第四组 `rerank=true`。
同一数据集、账号、权限范围、模型快照、索引代次和资源条件下运行，保存实际 JSON 结果后比较。
评测命令不会调用生成 API，答案正确率、拒答合理性与引用支持度需另外人工复核。服务降级会在结果中记录，不能把降级实验当原配置实验。

```powershell
knowledge-service eval --config evals/config.example.json --split dev --output artifacts/eval/run.json
knowledge-service compare --runs artifacts/eval/dense.json artifacts/eval/hybrid-rerank.json
```

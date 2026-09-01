RERANK_METHOD=llm python3 -m evaluator.local_evaluator \
  --dataset demo/live-demo-samples.jsonl \
  --output demo/live-results/result.json \
  --conversation-log demo/live-results/conversation.jsonl

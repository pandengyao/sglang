export PYTHONPATH=/ssd2/yaopandeng/work/sglang_normal/python/:$PYTHONPATH
SGLANG_USE_W4A8=1  SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path /ssd2/models/DeepSeek-R1-W4AFP8  \
    --tp 8 --trust-remote-code --host 0.0.0.0 --port 8005  --mem-fraction-static 0.85  --max-running-requests 1024 \
    --context-length 4096 --disable-cuda-graph --chunked-prefill-size 4096 --moe-dense-tp-size 1  --deepep-mode normal  \
    --moe-a2a-backend  deepep 2>&1 | tee normal.log
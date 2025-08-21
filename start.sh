export PYTHONPATH=/ssd2/yaopandeng/work/sglang_ll/python/:$PYTHONPATH
SGLANG_USE_W4A8=1 SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path /ssd2/models/DeepSeek-R1-W4AFP8 --tp 8 \
    --trust-remote-code --host 0.0.0.0 --port 8005 --context-length 8192 --moe-dense-tp-size 1 --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 16 --cuda-graph-bs 1 2 4 8 16 --max-running-requests 256 --disable-radix-cache --dp-size 8 \
    --enable-dp-attention --moe-a2a-backend deepep --deepep-mode low_latency 2>&1 | tee ll.log
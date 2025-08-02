"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    print("🚀 [LAUNCH_SERVER] Starting SGLang inference server...")
    
    # Set environment variables
    os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # 预设所有参数
    default_args = [
        "--model-path", "/ssd2/models/DeepSeek-R1-W4AFP8",
        "--context-length", "8192",
        "--tp-size", "8",
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--enable-ep-moe",
        "--cuda-graph-bs", "32",
        "--disable-radix-cache"
    ]
    
    server_args = prepare_server_args(default_args)
    # server_args = prepare_server_args(sys.argv[1:])

    try:
        print("🚀 [LAUNCH_SERVER] Calling launch_server...")
        print(f"🚀 [LAUNCH_SERVER] Server args prepared: server_args={server_args}")
            
        launch_server(server_args)
    finally:
        print("🚀 [LAUNCH_SERVER] Server shutdown, killing process tree...")
        kill_process_tree(os.getpid(), include_parent=False)

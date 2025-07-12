"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    print("🚀 [LAUNCH_SERVER] Starting SGLang inference server...")
    server_args = prepare_server_args(sys.argv[1:])
    print(f"🚀 [LAUNCH_SERVER] Server args prepared: server_args={server_args}")

    try:
        print("🚀 [LAUNCH_SERVER] Calling launch_server...")
        launch_server(server_args)
    finally:
        print("🚀 [LAUNCH_SERVER] Server shutdown, killing process tree...")
        kill_process_tree(os.getpid(), include_parent=False)

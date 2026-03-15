"""
scripts/verify_blender.py

Verifies the Blender bridge is reachable.
The agent never imports bpy — that lives inside Blender's own interpreter.
This script checks that the socket server launched by blender_server.py
is listening and responding.

Before running:
    1. Open Blender
    2. Run agent/tools/blender_server.py from Blender's scripting editor
       OR launch via: blender --python agent/tools/blender_server.py

Then run:
    uv run python scripts/verify_blender.py
"""

import json
import socket
import sys

HOST = "127.0.0.1"
PORT = 9999
TIMEOUT = 3.0


def check_socket_reachable() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=TIMEOUT):
            print(f"  [ok]  Blender socket server reachable at {HOST}:{PORT}")
            return True
    except ConnectionRefusedError:
        print(f"  [!!]  Connection refused at {HOST}:{PORT}")
        print("        Is Blender running blender_server.py?")
        return False
    except socket.timeout:
        print(f"  [!!]  Timed out connecting to {HOST}:{PORT}")
        return False


def check_ping() -> bool:
    """Send a real ping script and verify the response."""
    try:
        with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as sock:
            request = json.dumps({"id": "verify-ping", "script": "print('__pong__')"}) + "\n"
            sock.sendall(request.encode())
            raw = b""
            while b"\n" not in raw:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                raw += chunk
            response = json.loads(raw.split(b"\n")[0].decode())
            if response.get("success") and "__pong__" in (response.get("result") or ""):
                print("  [ok]  Ping script executed successfully inside Blender")
                return True
            else:
                print(f"  [!!]  Ping failed: {response}")
                return False
    except Exception as e:
        print(f"  [!!]  Ping error: {e}")
        return False


def check_bpy_available() -> bool:
    """Verify bpy is actually accessible inside Blender (not our venv)."""
    try:
        with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as sock:
            script = "import bpy; print(f'bpy {bpy.app.version_string}')"
            request = json.dumps({"id": "verify-bpy", "script": script}) + "\n"
            sock.sendall(request.encode())
            raw = b""
            while b"\n" not in raw:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                raw += chunk
            response = json.loads(raw.split(b"\n")[0].decode())
            if response.get("success"):
                version = (response.get("result") or "").strip()
                print(f"  [ok]  {version} accessible inside Blender")
                return True
            else:
                print(f"  [!!]  bpy not accessible: {response.get('error', '')[:100]}")
                return False
    except Exception as e:
        print(f"  [!!]  bpy check error: {e}")
        return False


def main() -> None:
    print("\nBlender bridge verification")
    print("─" * 50)
    print("  (bpy lives inside Blender — not this venv)")
    print()

    if not check_socket_reachable():
        print("\n  Start Blender and run agent/tools/blender_server.py first.\n")
        sys.exit(1)

    results = [
        check_ping(),
        check_bpy_available(),
    ]

    print("\n" + "─" * 50)
    if all(results):
        print("  Blender bridge ready.\n")
    else:
        print("  Bridge reachable but checks failed — see above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
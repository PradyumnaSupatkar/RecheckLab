"""Start a temporary headless Streamlit server and check HTTP startup."""
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    reports = ROOT / "outputs" / "v0.2" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "streamlit_smoke.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
             "--server.headless", "true", "--server.address", "127.0.0.1",
             "--server.port", str(port)], cwd=ROOT, stdout=log, stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Streamlit exited; see outputs/v0.2/reports/streamlit_smoke.log")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            break
                except (URLError, TimeoutError):
                    time.sleep(.2)
            else:
                raise RuntimeError("Streamlit health endpoint did not become ready within 30 seconds.")
            with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                if response.status != 200 or b"<html" not in response.read().lower():
                    raise RuntimeError("Streamlit did not serve the app shell.")
            print("PASS: Streamlit health and app shell HTTP 200; temporary server stopped.")
            print("Main prediction interaction is verified separately by Streamlit AppTest.")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Lightweight smoke test for the News Interpreter app.

Starts the Flask app in a subprocess, posts sample text to `/predict2`, and
checks the rendered HTML contains a summary and prediction labels.

Exits 0 on success, non-zero on failure.
"""
import os
import sys
import time
import signal
import subprocess

PORT = int(os.environ.get("PORT", "5002"))
URL_BASE = f"http://127.0.0.1:{PORT}"

SAMPLE_TEXT = (
    "This is a short test article. The quick brown fox jumps over the lazy dog. "
    "The service should summarize and predict labels based on the provided text."
)


def start_server():
    env = os.environ.copy()
    env["PORT"] = str(PORT)
    # Use the same python interpreter that's running this script
    proc = subprocess.Popen([sys.executable, os.path.join(os.getcwd(), "app.py")], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_for_up(timeout=12.0):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(URL_BASE + "/Input", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def run_test():
    try:
        import requests
    except Exception:
        print("requests not installed; please install with pip install requests", file=sys.stderr)
        return 2

    proc = start_server()
    try:
        up = wait_for_up()
        if not up:
            print("Server failed to start within timeout", file=sys.stderr)
            return 3

        r = requests.post(URL_BASE + "/predict2", data={"news": SAMPLE_TEXT}, timeout=10)
        if r.status_code != 200:
            print(f"Unexpected status: {r.status_code}", file=sys.stderr)
            return 4

        body = r.text
        checks = ["Summary", "Category"]
        cred_ok = ("Credibility" in body) or ("The news is" in body)
        if all(c in body for c in checks) and cred_ok:
            print("SMOKE TEST PASS")
            return 0
        else:
            print("SMOKE TEST FAIL: missing expected strings", file=sys.stderr)
            # dump short snippet for debugging
            print(body[:200], file=sys.stderr)
            return 5
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    code = run_test()
    sys.exit(code)

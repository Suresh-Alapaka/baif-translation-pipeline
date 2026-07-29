from flask import Flask, jsonify, render_template_string
import subprocess, sys, socket, time, os, threading

app = Flask(__name__)

# ⚠️ EDIT THIS: full path to your app_flask.py on your local machine
SCRIPT_PATH = r"C:\Users\OmPrakash\baif-translation-pipeline-lite\app_flask.py"          # Windows example
# SCRIPT_PATH = "/home/you/project/app_flask.py"  # macOS/Linux example

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 5000          # the port app_flask.py listens on
STARTUP_TIMEOUT = 180       # seconds to wait for models to load (adjust as needed)

_process = None
_lock = threading.Lock()


def is_port_open(host, port, timeout=1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_server(host, port, timeout):
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(host, port):
            return True
        time.sleep(1)
    return False


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Launcher</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 500px; margin: 80px auto; text-align: center; }
        button { padding: 14px 28px; font-size: 16px; cursor: pointer;
                 background: #1a73e8; color: white; border: none; border-radius: 8px; }
        button:disabled { background: #aaa; cursor: not-allowed; }
        #status { margin-top: 20px; font-size: 14px; }
        .error { color: #c62828; }
        .ok { color: #2e7d32; }
    </style>
</head>
<body>
    <h2>Start Translation App</h2>
    <button id="runBtn">Run Script</button>
    <div id="status"></div>

    <script>
        const btn = document.getElementById('runBtn');
        const statusDiv = document.getElementById('status');

        btn.addEventListener('click', async () => {
            btn.disabled = true;
            btn.textContent = 'Starting... (models are loading, this can take a while)';
            statusDiv.textContent = '';
            statusDiv.className = '';

            try {
                const response = await fetch('/run-script', { method: 'POST' });
                const data = await response.json();

                if (data.success) {
                    statusDiv.textContent = 'Started! Redirecting...';
                    statusDiv.className = 'ok';
                    window.location.href = 'http://localhost:5000';
                } else {
                    statusDiv.textContent = 'Failed: ' + data.message;
                    statusDiv.className = 'error';
                    btn.disabled = false;
                    btn.textContent = 'Run Script';
                }
            } catch (err) {
                statusDiv.textContent = 'Error contacting launcher: ' + err.message;
                statusDiv.className = 'error';
                btn.disabled = false;
                btn.textContent = 'Run Script';
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/run-script", methods=["POST"])
def run_script():
    global _process

    with _lock:
        # Already running? Just say so.
        if is_port_open(TARGET_HOST, TARGET_PORT):
            return jsonify({"success": True, "message": "App is already running"})

        # Start it only if we don't already have a live process handle
        if _process is None or _process.poll() is not None:
            if not os.path.isfile(SCRIPT_PATH):
                return jsonify({"success": False, "message": f"Script not found: {SCRIPT_PATH}"})
            try:
                _process = subprocess.Popen(
                    [sys.executable, SCRIPT_PATH],   # fixed arg list, no shell=True
                    cwd=os.path.dirname(SCRIPT_PATH),
                )
            except Exception as e:
                return jsonify({"success": False, "message": f"Failed to launch: {e}"})

    # Block this request until app_flask.py's server is actually up
    ready = wait_for_server(TARGET_HOST, TARGET_PORT, STARTUP_TIMEOUT)

    if ready:
        return jsonify({"success": True, "message": "App started successfully"})
    else:
        return jsonify({
            "success": False,
            "message": f"Timed out after {STARTUP_TIMEOUT}s waiting for the app to come up "
                       f"(model loading can be slow on CPU — try increasing STARTUP_TIMEOUT)"
        })


if __name__ == "__main__":
    print("✅ Launcher running at http://localhost:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)
from flask import Flask, jsonify, render_template_string
import subprocess, sys, socket, time, os, threading, tempfile
import psutil

app = Flask(__name__)

# Full path to your app_flask.py
SCRIPT_PATH = r"C:\Users\baif-translation-pipeline-lite\app_flask.py"

# Full path to your whisper-env virtual environment folder
# (the folder that CONTAINS Scripts\activate.bat)
VENV_PATH = r"C:\Users\hi\whisper-env"

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 5000          # the port app_flask.py listens on
STARTUP_TIMEOUT = 600       # seconds to wait for models to load (adjust as needed)

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


def find_pid_on_port(port):
    """
    Find the PID of whatever process is actually listening on `port`,
    regardless of whether the launcher started it. Covers app_flask.py
    being run manually in a terminal, or a previous launcher instance
    that lost its in-memory _process handle (e.g. after a reload).
    """
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN \
                    and conn.laddr and conn.laddr.port == port:
                return conn.pid
    except (psutil.AccessDenied, PermissionError):
        pass
    return None


def build_launch_command():
    """
    Build a command that runs app_flask.py using the whisper-env venv's
    own Python interpreter directly. Calling <venv>\\Scripts\\python.exe
    is functionally identical to running "activate" first and then
    "python app_flask.py" - it uses the venv's site-packages - but avoids
    spawning cmd.exe / a batch file entirely, so we can launch it with no
    visible console window at all.
    """
    script_dir = os.path.dirname(SCRIPT_PATH)

    if os.name == "nt":
        venv_python = os.path.join(VENV_PATH, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(VENV_PATH, "bin", "python")

    if not os.path.isfile(venv_python):
        raise FileNotFoundError(f"Venv python not found: {venv_python}")

    cmd = [venv_python, SCRIPT_PATH]
    return cmd, script_dir


def start_app_process():
    """
    Launch app_flask.py via the venv's python.exe, hidden, logging to a
    UTF-8 file. Returns (success: bool, message: str). Assumes caller
    already holds _lock.
    """
    global _process
    if not os.path.isfile(SCRIPT_PATH):
        return False, f"Script not found: {SCRIPT_PATH}"
    try:
        cmd, script_dir = build_launch_command()
        log_path = os.path.join(tempfile.gettempdir(), "app_flask_launcher.log")
        log_file = open(log_path, "a", encoding="utf-8")
        child_env = os.environ.copy()
        # Force UTF-8 stdout/stderr in the child process. Without this,
        # Python falls back to the Windows ANSI codepage (e.g. cp1252)
        # whenever stdout isn't attached to a real console - which
        # crashes on any emoji/unicode print() (like "✅ Whisper ready")
        # the instant we redirect output to a log file or hide the window.
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        popen_kwargs = dict(
            cwd=script_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
        if os.name == "nt":
            # Suppress the console window entirely.
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        _process = subprocess.Popen(cmd, **popen_kwargs)
        return True, "Launched"
    except Exception as e:
        return False, f"Failed to launch: {e}"


def stop_app_process(timeout=15):
    """
    Terminate app_flask.py and wait for TARGET_PORT to actually free up.
    Tries the tracked _process handle first (the normal case, when the
    launcher itself started it). If that handle is missing or stale -
    e.g. the script was started manually in a terminal, or a previous
    launcher instance lost track of it - falls back to finding whatever
    process is actually listening on TARGET_PORT and killing that by PID.
    Assumes caller already holds _lock. Returns (success: bool, message: str).
    """
    global _process
    if _process is not None and _process.poll() is None:
        try:
            _process.terminate()
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait()
        except Exception as e:
            return False, f"Failed to stop tracked process: {e}"
    _process = None

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_port_open(TARGET_HOST, TARGET_PORT):
            return True, "Stopped"

        # Something is still listening on the port but we have no
        # tracked handle for it (manual start, or a stale launcher
        # reference) - find its PID directly and kill that.
        pid = find_pid_on_port(TARGET_PORT)
        if pid:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                return False, (
                    f"Found PID {pid} on port {TARGET_PORT} but don't have "
                    f"permission to kill it. Close it manually."
                )
        time.sleep(0.5)
    return False, f"Port {TARGET_PORT} still open after stop attempt"


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


@app.after_request
def add_cors_headers(response):
    # app_flask.py's own page (served from a different port, so a
    # different origin) calls these endpoints via fetch() for its
    # restart button - allow that cross-origin request.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/run-script", methods=["POST"])
def run_script():
    with _lock:
        # Already running? Just say so.
        if is_port_open(TARGET_HOST, TARGET_PORT):
            return jsonify({"success": True, "message": "App is already running"})

        # Start it only if we don't already have a live process handle
        if _process is None or _process.poll() is not None:
            ok, msg = start_app_process()
            if not ok:
                return jsonify({"success": False, "message": msg})

    # Block this request until app_flask.py's server is actually up
    ready = wait_for_server(TARGET_HOST, TARGET_PORT, STARTUP_TIMEOUT)

    if ready:
        return jsonify({"success": True, "message": "App started successfully"})
    else:
        return jsonify({
            "success": False,
            "message": f"Timed out after {STARTUP_TIMEOUT}s waiting for the app to come up "
                       f"(model loading can be slow on CPU — try increasing STARTUP_TIMEOUT). "
                       f"Check the log for details: {os.path.join(tempfile.gettempdir(), 'app_flask_launcher.log')}"
        })


@app.route("/stop-script", methods=["POST"])
def stop_script():
    with _lock:
        ok, msg = stop_app_process()
    return jsonify({"success": ok, "message": msg})


@app.route("/restart-script", methods=["POST"])
def restart_script():
    with _lock:
        ok, msg = stop_app_process()
        if not ok:
            return jsonify({"success": False, "message": f"Stop failed: {msg}"})

        ok, msg = start_app_process()
        if not ok:
            return jsonify({"success": False, "message": msg})

    ready = wait_for_server(TARGET_HOST, TARGET_PORT, STARTUP_TIMEOUT)
    if ready:
        return jsonify({"success": True, "message": "App restarted successfully"})
    else:
        return jsonify({
            "success": False,
            "message": f"Timed out after {STARTUP_TIMEOUT}s waiting for the app to restart. "
                       f"Check the log: {os.path.join(tempfile.gettempdir(), 'app_flask_launcher.log')}"
        })


if __name__ == "__main__":
    print("✅ Launcher running at http://localhost:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)

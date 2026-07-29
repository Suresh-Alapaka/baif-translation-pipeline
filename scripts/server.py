from flask import Flask, request, jsonify, render_template
import subprocess
import os

app = Flask(__name__)

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "scripts", "my_script.py")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run-script", methods=["POST"])
def run_script():
    """
    Runs a server-side Python script safely using subprocess.run with a list
    of args (NOT shell=True), which avoids shell injection.
    """
    data = request.get_json(silent=True) or {}
    name_arg = data.get("name", "World")

    try:
        result = subprocess.run(
            ["python3", SCRIPT_PATH, name_arg],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return jsonify({
                "success": False,
                "message": "Script failed to execute.",
                "error": result.stderr.strip(),
            }), 500

        return jsonify({
            "success": True,
            "message": "Script ran successfully.",
            "output": result.stdout.strip(),
        }), 200

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "message": "Script timed out.",
            "error": "Execution exceeded 10 seconds.",
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Unexpected server error.",
            "error": str(e),
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)

"""HTTP routes and the local web server."""

import argparse

from flask import Flask, abort, jsonify, render_template, request, send_file

from .jobs import JobQueue
from .paths import OUTPUT_DIR


def create_app(output_dir=None, start_worker=True):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=16384, TRUSTED_HOSTS=["127.0.0.1", "localhost", "[::1]"])
    queue = JobQueue(output_dir or OUTPUT_DIR, start_worker=start_worker)
    app.extensions["job_queue"] = queue

    @app.before_request
    def local_requests():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                abort(403)
            if not request.is_json:
                abort(415)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/jobs")
    def jobs():
        return jsonify(queue.list_jobs())

    @app.post("/api/jobs")
    def submit():
        try:
            return jsonify(queue.submit(request.get_json())), 202
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/jobs/<job_id>/log")
    def job_log(job_id):
        if job_id not in {j["id"] for j in queue.list_jobs()}:
            abort(404)
        path = queue.jobs_dir / job_id / "run.log"
        if not path.exists():
            return jsonify(text="Обработка ещё не началась.")
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 24000))
            return jsonify(text=log.read().decode("utf-8", errors="replace"))

    @app.get("/api/results")
    def results():
        return jsonify([{**r, "files": list(r["files"])} for r in queue.results()])

    def find_result(result_id):
        result = next((r for r in queue.results() if r["id"] == result_id), None)
        if not result:
            abort(404)
        return result

    @app.get("/api/results/<result_id>")
    def result_detail(result_id):
        result = find_result(result_id)
        texts = {
            kind: path.read_text(encoding="utf-8")
            for kind, path in result["files"].items()
            if kind in {"chapters", "transcript"}
        }
        return jsonify(**{**result, "files": list(result["files"])}, **texts)

    @app.get("/api/results/<result_id>/download/<kind>")
    def download(result_id, kind):
        path = find_result(result_id)["files"].get(kind)
        if not path:
            abort(404)
        return send_file(path, as_attachment=True)

    return app


def main():
    from waitress import serve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    serve(create_app(), host="127.0.0.1", port=args.port, threads=4)


if __name__ == "__main__":
    main()

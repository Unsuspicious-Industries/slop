import logging
import platform
import signal
import sys
import traceback

import torch
import os

from server.inference.runner import InferenceRunner
from shared.protocol.messages import (
    AttachJobRequest,
    CleanupRequest,
    DatasetStatsRequest,
    DecodeRequest,
    EmbedRequest,
    EncodeRequest,
    ErrorResponse,
    InferenceRequest,
    JobResult,
    KillJobRequest,
    ListJobsRequest,
    MessageKind,
    ProgressResponse,
    Request,
    Response,
    ServerInfo,
    TrainRequest,
)
from shared.protocol.wire import read_message, write_message


logging.basicConfig(
    level=logging.INFO,
    format="[SERVER] %(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class ServerDaemon:
    def __init__(self) -> None:
        # Ensure only one daemon instance runs per remote path. Create a PID
        # lock file in the current working directory to prevent duplicate
        # server processes from being spawned (duplicate processes lead to
        # multiple models loaded into GPU memory).
        self.lock_path = os.path.join(os.getcwd(), ".server.lock")
        try:
            if os.path.exists(self.lock_path):
                try:
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        pid_text = f.read().strip()
                        pid = int(pid_text) if pid_text else None
                except Exception:
                    pid = None
                if pid:
                    try:
                        # signal 0 checks for process existence without killing
                        os.kill(pid, 0)
                        logger.info("another server (pid=%s) appears to be running — exiting", pid)
                        raise SystemExit(0)
                    except OSError:
                        # Stale lock, remove
                        try:
                            os.remove(self.lock_path)
                        except Exception:
                            pass
        except Exception:
            # If anything goes wrong in lock handling, continue; we still want
            # the daemon to run rather than fail hard.
            pass

        # Write our PID to the lock file
        try:
            with open(self.lock_path, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass

        self.runner = InferenceRunner()
        self.running = True
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, _frame) -> None:
        logger.info("received signal %s", signum)
        self.running = False
        self.runner.clear()
        try:
            if hasattr(self, "lock_path") and os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except Exception:
            pass

    def handle_info(self, req: Request) -> ServerInfo:
        cuda_version = torch.version.cuda or "N/A"
        info = ServerInfo(
            job_id=req.job_id,
            hostname=platform.node(),
            cuda_version=cuda_version if torch.cuda.is_available() else "N/A",
            torch_version=torch.__version__,
            capabilities=["sample", "render", "batch", "encode", "probe"],
            loaded_models=[self.runner.current_model_id] if self.runner.current_model_id else [],
        )
        if torch.cuda.is_available():
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_memory_mb = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        return info

    def process_request(self, data: dict) -> Response:
        try:
            kind = data.get("kind")
            if kind == MessageKind.PING.value:
                req = Request.from_dict(data)
                return Response(kind=MessageKind.PONG.value, job_id=req.job_id)
            if kind == MessageKind.SERVER_INFO.value:
                return self.handle_info(Request.from_dict(data))
            if kind == MessageKind.INFERENCE.value:
                req = InferenceRequest.from_dict(data)
                logger.info("sample/probe request model=%s", req.model_id)
                return self.runner.run(req)
            if kind == MessageKind.ENCODE.value:
                req = EncodeRequest.from_dict(data)
                if req.modality != "text":
                    return ErrorResponse(job_id=req.job_id, error=f"unsupported encode modality: {req.modality}")
                return self.runner.encode_prompt(req)
            if kind == MessageKind.EMBED.value:
                req = EmbedRequest.from_dict(data)
                return self.runner.embed_prompt(req)
            if kind == MessageKind.DECODE.value:
                req = DecodeRequest.from_dict(data)
                return self.runner.decode_latents(req)
            if kind == MessageKind.CLEANUP.value:
                req = CleanupRequest.from_dict(data)
                stats = self.runner.cleanup(clear_model=req.clear_model)
                # Return a JobResult so the client can read a payload dict with stats.
                return JobResult(
                    job_id=req.job_id,
                    elapsed_s=0.0,
                    request_kind=MessageKind.CLEANUP.value,
                    payload=stats,
                )
            if kind == MessageKind.TRAIN.value:
                req = TrainRequest.from_dict(data)
                return self.start_training(req)
            if kind == MessageKind.JOB_LIST.value:
                req = ListJobsRequest.from_dict(data)
                return self.list_jobs(req)
            if kind == MessageKind.JOB_ATTACH.value:
                req = AttachJobRequest.from_dict(data)
                return self.attach_job(req)
            if kind == MessageKind.JOB_KILL.value:
                req = KillJobRequest.from_dict(data)
                return self.kill_job(req)
            if kind == MessageKind.DATASET_STATS.value:
                req = DatasetStatsRequest.from_dict(data)
                return self.dataset_stats(req)
            if kind == MessageKind.SHUTDOWN.value:
                self.running = False
                self.runner.clear()
                return Response(kind=MessageKind.INFO.value, job_id=data.get("job_id", ""), elapsed_s=0.0)
            return ErrorResponse(job_id=data.get("job_id", ""), error=f"unknown message kind: {kind}")
        except Exception as exc:
            logger.error("request failed: %s", exc)
            traceback.print_exc(file=sys.stderr)
            return ErrorResponse(
                job_id=data.get("job_id", ""),
                error=str(exc),
                traceback=traceback.format_exc(),
            )

    def _jobs_root(self) -> str:
        return os.environ.get("SLOP_JOBS_DIR") or os.path.join(os.path.expanduser("~"), ".slop", "jobs")

    def _job_dir(self, job_id: str) -> str:
        return os.path.join(self._jobs_root(), job_id)

    def start_training(self, req: TrainRequest) -> JobResult:
        """Start an autonomous training job.

        The job runs in a detached worker process and continues even if the
        client disconnects. Use JOB_LIST / JOB_ATTACH / JOB_KILL to manage.
        """
        import json
        import subprocess
        import time
        from pathlib import Path

        job_id = req.job_id
        job_dir = self._job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)

        spec_path = os.path.join(job_dir, "spec.json")
        status_path = os.path.join(job_dir, "status.json")
        progress_path = os.path.join(job_dir, "progress.jsonl")
        log_path = os.path.join(job_dir, "worker.log")
        pid_path = os.path.join(job_dir, "pid")

        spec = {
            "job_id": job_id,
            "kind": "train",
            "model_id": req.model_id,
            "manifest_path": req.manifest_path,
            "output_dir": req.output_dir,
            "batch_size": req.batch_size,
            "epochs": req.epochs,
            "learning_rate": req.learning_rate,
            "lora_rank": req.lora_rank,
            "save_every": req.save_every,
            "created_at": time.time(),
        }
        Path(spec_path).write_text(json.dumps(spec, indent=2), encoding="utf-8")

        # Clear previous progress
        try:
            if os.path.exists(progress_path):
                os.remove(progress_path)
        except Exception:
            pass

        # Initial status
        Path(status_path).write_text(
            json.dumps({"job_id": job_id, "state": "starting", "updated_at": time.time()}, indent=2),
            encoding="utf-8",
        )

        python = sys.executable
        cmd = [python, "-m", "distill.train_worker", "--job-dir", job_dir]

        with open(log_path, "ab") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=logf,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=os.getcwd(),
            )

        Path(pid_path).write_text(str(proc.pid), encoding="utf-8")
        logger.info("started autonomous training job %s pid=%s", job_id, proc.pid)

        return JobResult(
            job_id=job_id,
            elapsed_s=0.0,
            request_kind=MessageKind.TRAIN.value,
            payload={
                "job_id": job_id,
                "job_dir": job_dir,
                "pid": proc.pid,
                "status_path": status_path,
                "progress_path": progress_path,
                "log_path": log_path,
            },
        )

    def list_jobs(self, req: ListJobsRequest) -> JobResult:
        import json
        import time
        from pathlib import Path
        root = self._jobs_root()
        jobs = []
        if os.path.isdir(root):
            for name in sorted(os.listdir(root), reverse=True):
                if len(jobs) >= max(1, req.limit):
                    break
                job_dir = os.path.join(root, name)
                status_path = os.path.join(job_dir, "status.json")
                spec_path = os.path.join(job_dir, "spec.json")
                if not os.path.isdir(job_dir) or not os.path.exists(spec_path):
                    continue
                status = {}
                if os.path.exists(status_path):
                    try:
                        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
                    except Exception:
                        status = {}
                try:
                    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
                except Exception:
                    spec = {}
                jobs.append({
                    "job_id": name,
                    "kind": spec.get("kind", ""),
                    "state": status.get("state", "unknown"),
                    "updated_at": status.get("updated_at", 0.0),
                    "model_id": spec.get("model_id", ""),
                    "output_dir": spec.get("output_dir", ""),
                })
        return JobResult(
            job_id=req.job_id,
            elapsed_s=0.0,
            request_kind=MessageKind.JOB_LIST.value,
            payload={"jobs": jobs},
        )

    def attach_job(self, req: AttachJobRequest) -> JobResult:
        import json
        import time
        from pathlib import Path
        job_dir = self._job_dir(req.target_job_id)
        status_path = os.path.join(job_dir, "status.json")
        progress_path = os.path.join(job_dir, "progress.jsonl")
        log_path = os.path.join(job_dir, "worker.log")

        if not os.path.isdir(job_dir):
            return JobResult(job_id=req.job_id, request_kind=MessageKind.JOB_ATTACH.value, payload={"error": "job not found"})

        status = {}
        if os.path.exists(status_path):
            try:
                status = json.loads(Path(status_path).read_text(encoding="utf-8"))
            except Exception:
                status = {}

        lines = []
        next_since = req.since_line
        if os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < req.since_line:
                        continue
                    if len(lines) >= max(1, req.max_lines):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        continue
                    next_since = i + 1

        # small log tail for debugging
        log_tail = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, "rb") as lf:
                    lf.seek(0, os.SEEK_END)
                    size = lf.tell()
                    lf.seek(max(0, size - 4096), os.SEEK_SET)
                    log_tail = lf.read().decode("utf-8", errors="replace")
            except Exception:
                log_tail = ""

        return JobResult(
            job_id=req.job_id,
            elapsed_s=0.0,
            request_kind=MessageKind.JOB_ATTACH.value,
            payload={
                "job_id": req.target_job_id,
                "status": status,
                "progress": lines,
                "next_since_line": next_since,
                "log_tail": log_tail,
            },
        )

    def kill_job(self, req: KillJobRequest) -> JobResult:
        import json
        import time
        from pathlib import Path
        job_dir = self._job_dir(req.target_job_id)
        pid_path = os.path.join(job_dir, "pid")
        status_path = os.path.join(job_dir, "status.json")
        if not os.path.exists(pid_path):
            return JobResult(job_id=req.job_id, request_kind=MessageKind.JOB_KILL.value, payload={"error": "pid not found"})
        pid_txt = Path(pid_path).read_text(encoding="utf-8").strip()
        try:
            pid = int(pid_txt)
        except Exception:
            return JobResult(job_id=req.job_id, request_kind=MessageKind.JOB_KILL.value, payload={"error": f"bad pid: {pid_txt}"})

        sig = signal.SIGTERM if req.signal == "term" else signal.SIGKILL
        try:
            os.kill(pid, sig)
            # update status best-effort
            if os.path.exists(status_path):
                try:
                    st = json.loads(Path(status_path).read_text(encoding="utf-8"))
                except Exception:
                    st = {}
            else:
                st = {}
            st.update({"state": "killed", "updated_at": time.time(), "killed_by": "client", "signal": req.signal})
            Path(status_path).write_text(json.dumps(st, indent=2), encoding="utf-8")
            return JobResult(job_id=req.job_id, request_kind=MessageKind.JOB_KILL.value, payload={"ok": True, "pid": pid})
        except Exception as e:
            return JobResult(job_id=req.job_id, request_kind=MessageKind.JOB_KILL.value, payload={"ok": False, "error": str(e), "pid": pid})

    def dataset_stats(self, req: DatasetStatsRequest) -> JobResult:
        import json
        from pathlib import Path

        manifest = Path(req.manifest_path)
        if not manifest.exists():
            return JobResult(
                job_id=req.job_id,
                request_kind=MessageKind.DATASET_STATS.value,
                payload={"error": f"manifest not found: {req.manifest_path}"},
            )

        total_lines = 0
        ok_images = 0
        missing_images = 0
        prompts = set()
        teachers = {}
        image_sizes = []
        bytes_total = 0
        sample_prompts = []

        # Only open a limited number of images for size stats
        max_open = max(0, int(req.sample_images))
        opened = 0

        try:
            from PIL import Image
        except Exception:
            Image = None

        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if total_lines >= int(req.max_records):
                    break
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                p = str(rec.get("prompt", ""))
                if p:
                    prompts.add(p)
                    if len(sample_prompts) < 8:
                        sample_prompts.append(p)

                t = str(rec.get("teacher", ""))
                if t:
                    teachers[t] = teachers.get(t, 0) + 1

                final_path = rec.get("final_path")
                if not final_path:
                    missing_images += 1
                    continue
                img_path = Path(final_path)
                if not img_path.is_absolute():
                    # manifest paths are typically relative to repo root
                    img_path = (manifest.parent / img_path).resolve()

                if not img_path.exists():
                    missing_images += 1
                    continue

                ok_images += 1
                try:
                    bytes_total += img_path.stat().st_size
                except Exception:
                    pass

                if Image is not None and opened < max_open:
                    try:
                        with Image.open(img_path) as im:
                            image_sizes.append({"w": int(im.size[0]), "h": int(im.size[1])})
                        opened += 1
                    except Exception:
                        pass

        payload = {
            "manifest_path": str(manifest),
            "records_read": total_lines,
            "images_ok": ok_images,
            "images_missing": missing_images,
            "unique_prompts": len(prompts),
            "teachers": teachers,
            "bytes_total": bytes_total,
            "sample_prompts": sample_prompts,
            "sample_image_sizes": image_sizes,
        }

        return JobResult(job_id=req.job_id, request_kind=MessageKind.DATASET_STATS.value, payload=payload)
    
    def _send_progress(self, job_id: str, step: int, epoch: int, loss: float, message: str):
        """Send a progress update to the client."""
        import json
        try:
            progress = ProgressResponse(
                job_id=job_id,
                step=step,
                epoch=epoch,
                loss=loss,
                message=message,
            )
            # Write to stderr for progress (separate from main response channel)
            sys.stderr.write(f"[PROGRESS] {json.dumps(progress.to_dict())}\n")
            sys.stderr.flush()
        except Exception as e:
            logger.warning(f"Failed to send progress: {e}")

    def run(self) -> None:
        logger.info("server ready")
        input_stream = sys.stdin.buffer
        output_stream = sys.stdout.buffer
        while self.running:
            try:
                data = read_message(input_stream)
                if data is None:
                    break
                response = self.process_request(data)
                write_message(output_stream, response.to_dict())
                output_stream.flush()
            except Exception as exc:
                logger.critical("fatal loop error: %s", exc)
                traceback.print_exc(file=sys.stderr)
                break
        self.runner.clear()
        try:
            if hasattr(self, "lock_path") and os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except Exception:
            pass


if __name__ == "__main__":
    ServerDaemon().run()

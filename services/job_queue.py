import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional


@dataclass
class Job:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    filename: Optional[str] = None
    status: str = "queued"
    error: Optional[str] = None
    indexed_records: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class JobQueue:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def submit(self, job_type: str, filename: Optional[str], task_fn: Callable[[], int]) -> Job:
        job = Job(type=job_type, filename=filename)
        with self._jobs_lock:
            self._jobs[job.id] = job
        self._queue.put((job.id, task_fn))
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def _worker(self):
        while True:
            job_id, task_fn = self._queue.get()
            with self._jobs_lock:
                job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue

            job.status = "running"
            try:
                result = task_fn()
                job.status = "complete"
                job.indexed_records = result
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.completed_at = datetime.now(timezone.utc)
                self._queue.task_done()


job_queue = JobQueue()

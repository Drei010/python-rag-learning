import threading
import time
import unittest

from services.job_queue import Job, JobQueue


class TestJobQueue(unittest.TestCase):
    def setUp(self):
        self.queue = JobQueue()
        self.queue.start()

    def test_job_starts_queued_then_completes(self):
        """Job starts as queued, transitions to running, then complete."""
        started_event = threading.Event()
        proceed_event = threading.Event()

        def task_fn() -> int:
            started_event.set()
            proceed_event.wait(timeout=5)
            return 42

        job = self.queue.submit("index", "test.pdf", task_fn)
        self.assertEqual(job.status, "queued")

        # Wait for the worker to pick up the job
        started_event.wait(timeout=5)
        self.assertEqual(job.status, "running")

        # Let the task complete
        proceed_event.set()

        # Wait for completion
        self.queue._queue.join()
        self.assertEqual(job.status, "complete")
        self.assertEqual(job.indexed_records, 42)
        self.assertIsNotNone(job.completed_at)

    def test_failed_task_sets_status_failed(self):
        """Failed task sets status to failed with error message."""
        def task_fn() -> int:
            raise ValueError("Something went wrong")

        job = self.queue.submit("index", "bad.pdf", task_fn)
        self.queue._queue.join()

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "Something went wrong")
        self.assertIsNotNone(job.completed_at)

    def test_jobs_process_sequentially(self):
        """Multiple jobs process sequentially (second job waits for first)."""
        order = []
        lock = threading.Lock()

        def make_task(name: str):
            def task_fn() -> int:
                with lock:
                    order.append(f"{name}_start")
                time.sleep(0.05)
                with lock:
                    order.append(f"{name}_end")
                return 1
            return task_fn

        self.queue.submit("index", "a.pdf", make_task("first"))
        self.queue.submit("index", "b.pdf", make_task("second"))

        self.queue._queue.join()

        self.assertEqual(order, ["first_start", "first_end", "second_start", "second_end"])

    def test_get_job_returns_correct_job(self):
        """get_job returns correct job by ID."""
        def task_fn() -> int:
            return 10

        job = self.queue.submit("reindex", None, task_fn)
        self.queue._queue.join()

        retrieved = self.queue.get_job(job.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, job.id)
        self.assertEqual(retrieved.type, "reindex")
        self.assertEqual(retrieved.status, "complete")
        self.assertEqual(retrieved.indexed_records, 10)

    def test_get_job_returns_none_for_unknown_id(self):
        """get_job returns None for unknown job ID."""
        self.assertIsNone(self.queue.get_job("nonexistent-id"))

    def test_get_all_jobs_reverse_chronological(self):
        """get_all_jobs returns jobs in reverse chronological order."""
        def task_fn() -> int:
            return 1

        job1 = self.queue.submit("index", "a.pdf", task_fn)
        self.queue._queue.join()
        time.sleep(0.01)  # Ensure different timestamps

        job2 = self.queue.submit("index", "b.pdf", task_fn)
        self.queue._queue.join()
        time.sleep(0.01)

        job3 = self.queue.submit("reindex", None, task_fn)
        self.queue._queue.join()

        all_jobs = self.queue.get_all_jobs()
        self.assertEqual(len(all_jobs), 3)
        # Most recent first
        self.assertEqual(all_jobs[0].id, job3.id)
        self.assertEqual(all_jobs[1].id, job2.id)
        self.assertEqual(all_jobs[2].id, job1.id)


if __name__ == "__main__":
    unittest.main()

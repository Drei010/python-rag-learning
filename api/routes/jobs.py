from fastapi import APIRouter, HTTPException, status

from models.schemas import JobListResponse, JobStatusResponse
from services.job_queue import job_queue


router = APIRouter(tags=["jobs"])


@router.get("/jobs")
def list_jobs() -> JobListResponse:
    jobs = job_queue.get_all_jobs()
    return JobListResponse(
        jobs=[
            JobStatusResponse(
                job_id=job.id,
                status=job.status,
                type=job.type,
                filename=job.filename,
                error=job.error,
                indexed_records=job.indexed_records,
                created_at=job.created_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
            )
            for job in jobs
        ]
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> JobStatusResponse:
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        type=job.type,
        filename=job.filename,
        error=job.error,
        indexed_records=job.indexed_records,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )

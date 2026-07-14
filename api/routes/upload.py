from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from typing import Optional

from models.schemas import ReindexAcceptedResponse, UploadAcceptedResponse
from services import embed_service
from services.file_service import (
    delete_file_from_storage,
    file_exists,
    is_supported_file,
    sanitize_filename,
    save_upload_file,
)
from services.job_queue import job_queue
from services.metadata_service import file_metadata_store


router = APIRouter(tags=["files"])


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_file(
    file: UploadFile = File(...),
    uploaded_by: Optional[str] = Form(default=None),
) -> UploadAcceptedResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must include a filename.",
        )

    filename = sanitize_filename(file.filename)
    if not is_supported_file(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Excel, PDF, and PowerPoint (.pptx/.pptm) files are supported.",
        )

    if file_exists(filename):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{filename} already exists in file storage.",
        )

    try:
        location = await save_upload_file(file, filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        ) from exc

    file_metadata_store.record_upload(filename, uploaded_by=uploaded_by)

    def task_fn() -> int:
        try:
            return embed_service.index_file(filename)
        except Exception:
            delete_file_from_storage(filename, missing_ok=True)
            raise

    job = job_queue.submit("index", filename, task_fn)

    return UploadAcceptedResponse(
        job_id=job.id,
        filename=filename,
        path=location,
        status="queued",
    )


@router.post("/reindex", status_code=status.HTTP_202_ACCEPTED)
def reindex_files() -> ReindexAcceptedResponse:
    def task_fn() -> int:
        return embed_service.refresh_vector_store()

    job = job_queue.submit("reindex", None, task_fn)

    return ReindexAcceptedResponse(
        job_id=job.id,
        status="queued",
    )

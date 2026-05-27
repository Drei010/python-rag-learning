from fastapi import APIRouter, File, HTTPException, UploadFile, status

from core.config import settings
from models.schemas import FileUploadResponse, ReindexResponse
from services import embed_service
from services.file_service import (
    get_supported_files,
    is_supported_file,
    sanitize_filename,
    save_upload_file,
)


router = APIRouter(tags=["files"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)) -> FileUploadResponse:
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

    destination = settings.data_dir / filename
    if destination.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{filename} already exists in the data folder.",
        )

    await save_upload_file(file, destination)

    try:
        indexed_records = embed_service.refresh_vector_store()
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not index uploaded file: {exc}",
        ) from exc

    return FileUploadResponse(
        filename=filename,
        path=str(destination),
        indexed_records=indexed_records,
    )


@router.post("/reindex")
def reindex_files() -> ReindexResponse:
    indexed_records = embed_service.refresh_vector_store()
    files = [path.name for path in get_supported_files()]
    return ReindexResponse(indexed_records=indexed_records, files=files)

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from models.schemas import FileUploadResponse, ReindexResponse
from services import embed_service
from services.file_service import (
    delete_file_from_storage,
    file_exists,
    is_supported_file,
    list_stored_files,
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

    try:
        indexed_records = embed_service.refresh_vector_store()
    except Exception as exc:
        delete_file_from_storage(filename, missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not index uploaded file: {exc}",
        ) from exc

    return FileUploadResponse(
        filename=filename,
        path=location,
        indexed_records=indexed_records,
    )


@router.post("/reindex")
def reindex_files() -> ReindexResponse:
    indexed_records = embed_service.refresh_vector_store()
    files = list_stored_files()
    return ReindexResponse(indexed_records=indexed_records, files=files)

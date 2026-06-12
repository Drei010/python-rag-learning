from fastapi import APIRouter, HTTPException, status

from models.schemas import FileListResponse
from services.file_service import get_storage_destination, list_stored_files


router = APIRouter(tags=["files"])


@router.get("/files", response_model=FileListResponse)
def list_files() -> FileListResponse:
    try:
        destination, location = get_storage_destination()
        files = list_stored_files()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not list stored files: {exc}",
        ) from exc

    return FileListResponse(
        destination=destination,
        location=location,
        files=files,
    )

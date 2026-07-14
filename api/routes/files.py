from fastapi import APIRouter, HTTPException, status

from models.schemas import FileInfo, FileListResponse
from services.file_service import get_storage_destination, list_stored_file_details


router = APIRouter(tags=["files"])


@router.get("/files", response_model=FileListResponse)
def list_files() -> FileListResponse:
    try:
        destination, location = get_storage_destination()
        file_details = list_stored_file_details()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not list stored files: {exc}",
        ) from exc

    return FileListResponse(
        destination=destination,
        location=location,
        files=[
            FileInfo(
                filename=detail.filename,
                uploaded_by=detail.uploaded_by,
                uploaded_at=detail.uploaded_at,
            )
            for detail in file_details
        ],
    )

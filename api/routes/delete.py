from fastapi import APIRouter, HTTPException, status

from models.schemas import DeleteFileResponse
from services import embed_service
from services.file_service import (
    delete_file_from_storage,
    file_exists,
    sanitize_filename,
)


router = APIRouter(tags=["files"])


@router.delete(
    "/files/{filename}",
    status_code=status.HTTP_200_OK,
    response_model=DeleteFileResponse,
)
def delete_file(filename: str) -> DeleteFileResponse:
    filename = sanitize_filename(filename)

    if not file_exists(filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{filename} not found.",
        )

    try:
        delete_file_from_storage(filename)

        indexed_records = embed_service.refresh_vector_store()

        return DeleteFileResponse(
            filename=filename,
            indexed_records=indexed_records,
            message="File deleted successfully.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not delete file: {exc}",
        ) from exc

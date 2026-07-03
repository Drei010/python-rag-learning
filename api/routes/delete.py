from fastapi import APIRouter, HTTPException, status

from models.schemas import DeleteFilesResponse, DeleteFilesRequest
from services import embed_service
from services.file_service import (
    delete_file_from_storage,
    file_exists,
    sanitize_filename,
)


router = APIRouter(tags=["files"])

@router.delete(
    "/files",
    status_code=status.HTTP_200_OK,
    response_model=DeleteFilesResponse,
)
def delete_files(request: DeleteFilesRequest) -> DeleteFilesResponse:
    deleted = []
    not_found = []

    try:
        for filename in request.filenames:
            filename = sanitize_filename(filename)

            if file_exists(filename):
                delete_file_from_storage(filename)
                embed_service.remove_file(filename)
                deleted.append(filename)
            else:
                not_found.append(filename)

        indexed_records = len(embed_service.documents)

        return DeleteFilesResponse(
            deleted=deleted,
            not_found=not_found,
            indexed_records=indexed_records,
            message="Files processed successfully.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not delete files: {exc}",
        ) from exc
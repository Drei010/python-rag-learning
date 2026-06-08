from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str


class FileUploadResponse(BaseModel):
    filename: str
    path: str
    indexed_records: int


class ReindexResponse(BaseModel):
    indexed_records: int
    files: List[str]

class DeleteFileResponse(BaseModel):
    filename: str
    indexed_records: int
    message: str

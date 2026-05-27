from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Body, Cookie, HTTPException, Query, Response, status

from models.schemas import ChatRequest, ChatResponse
from services.rag_service import answer_question


router = APIRouter(tags=["chat"])


@router.post("/chat")
def ask_question(
    response: Response,
    request: Optional[ChatRequest] = Body(default=None),
    question: Optional[str] = Query(default=None, min_length=1),
    session_id: Optional[str] = Query(default=None, min_length=1),
    cookie_session_id: Optional[str] = Cookie(default=None, alias="ai_session_id"),
) -> ChatResponse:
    user_question = question or (request.question if request else None)
    user_session_id = (
        session_id
        or (request.session_id if request else None)
        or cookie_session_id
        or str(uuid4())
    )

    if not user_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a question in the JSON body or as a query parameter.",
        )

    answer = answer_question(user_question, user_session_id)
    response.set_cookie(
        key="ai_session_id",
        value=user_session_id,
        httponly=True,
        samesite="lax",
    )
    return ChatResponse(
        session_id=user_session_id,
        question=user_question,
        answer=answer,
    )

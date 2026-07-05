from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.chat import router as chat_router
from api.routes.delete import router as delete_router
from api.routes.files import router as files_router
from api.routes.jobs import router as jobs_router
from api.routes.upload import router as upload_router
from services.job_queue import job_queue
from services.rag_service import answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    job_queue.start()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router, prefix="/ai")
app.include_router(upload_router, prefix="/ai")
app.include_router(files_router, prefix="/ai")
app.include_router(delete_router, prefix="/ai")
app.include_router(jobs_router, prefix="/ai")


if __name__ == "__main__":
    while True:
        print("\n\n------------------------------")
        question = input("Ask your question (q to quit) : ")
        print("\n\n")
        if question == "q":
            break

        result = answer_question(question, "cli")
        print(result)

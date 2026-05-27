from fastapi import FastAPI

from api.routes.chat import router as chat_router
from api.routes.upload import router as upload_router
from services.rag_service import answer_question


app = FastAPI()
app.include_router(chat_router, prefix="/ai")
app.include_router(upload_router, prefix="/ai")


if __name__ == "__main__":
    while True:
        print("\n\n------------------------------")
        question = input("Ask your question (q to quit) : ")
        print("\n\n")
        if question == "q":
            break

        result = answer_question(question, "cli")
        print(result)

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="qwen3:14b")

template = """
You are an expert assistant. Answer the user's question clearly and concisely.

Context: 
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n------------------------------")
    question = input("Ask your question (q to quit) : ")
    print("\n\n")
    if question == "q":
        break

    context = retriever.invoke(question)
    print(f"DEBUG: Found {len(context)} relevant documents.")
    for d in context:
        print(f"DEBUG Content: {d.page_content[:50]}...")

    result = chain.invoke({"context": context, "question": question})
    print(result)
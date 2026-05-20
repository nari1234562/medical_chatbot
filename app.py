from flask import Flask, render_template, request, session

from src.helper import download_bedrock_embeddings

from langchain_community.vectorstores import Pinecone as LangchainPinecone

from langchain_aws import ChatBedrock

from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from src.prompt import system_prompt

import os

load_dotenv()

app = Flask(__name__)

app.secret_key = "medical_chatbot_secret"

# Pinecone key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Bedrock embeddings
embeddings = download_bedrock_embeddings()

index_name = "medicalbot"

# Vector DB
docsearch = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Bedrock Claude model
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0.4,
        "max_tokens": 500
    }
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Chat History:\n{history}\n\nUser: {input}")
])

# QA chain
question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# Home
@app.route("/")
def index():

    session["chat_history"] = []

    return render_template("chat.html")


# Chat endpoint
@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("User:", msg)

    history = session.get("chat_history", [])

    formatted_history = ""

    for h in history:

        formatted_history += (
            f"User: {h['user']}\n"
            f"Bot: {h['bot']}\n"
        )

    response = rag_chain.invoke({
        "input": msg,
        "history": formatted_history
    })

    answer = response["answer"]

    history.append({
        "user": msg,
        "bot": answer
    })

    session["chat_history"] = history[-10:]

    print("Bot:", answer)

    return str(answer)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
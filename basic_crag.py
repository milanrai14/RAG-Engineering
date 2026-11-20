
import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()


# ==========================================
# 2. INITIALIZE MODELS
# ==========================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ==========================================
# 3. LOAD AND SPLIT DOCUMENTS
# ==========================================

def load_documents():

    documents = []

    folder_path = "./documents"

    for filename in os.listdir(folder_path):

        if filename.endswith(".pdf"):

            file_path = os.path.join(
                folder_path,
                filename
            )

            loader = PyPDFLoader(file_path)

            documents.extend(loader.load())

    if not documents:
        raise ValueError("No PDF documents found!")

    print(f"Loaded {len(documents)} pages.")

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ==========================================
# 4. CREATE VECTOR DATABASE
# ==========================================

def create_vector_store():

    vector_store = Chroma(
        collection_name="crag_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # Avoid indexing the same PDFs every time
    if vector_store._collection.count() == 0:

        documents = load_documents()

        chunks = split_documents(documents)

        vector_store.add_documents(chunks)

        print("Documents added to ChromaDB.")

    else:

        print("Existing ChromaDB loaded.")

    return vector_store


vector_store = create_vector_store()


# ==========================================
# 5. INITIALIZE WEB SEARCH
# ==========================================

web_search_tool = DuckDuckGoSearchRun()


# ==========================================
# 6. DOCUMENT GRADER
# ==========================================

class GradeDocument(BaseModel):

    binary_score: Literal["yes", "no"] = Field(
        description="Is the document relevant to the question?"
    )


grader_llm = llm.with_structured_output(GradeDocument)


grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a document relevance evaluator.

            Your task is to determine whether the
            retrieved document contains information
            relevant to answering the question.

            Return:
            - yes: if the document is relevant.
            - no: if the document is irrelevant.

            Be strict when evaluating relevance.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Retrieved Document:
            {document}
            """
        )
    ]
)

document_grader = grader_prompt | grader_llm


# ==========================================
# 7. GENERATION PROMPT
# ==========================================

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the user's question using only
            the provided context.

            If the context does not contain enough
            information, clearly state that.

            Do not invent information.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Context:
            {context}

            Generate a clear and accurate answer.
            """
        )
    ]
)

generator = generation_prompt | llm


# ==========================================
# 8. LANGGRAPH STATE
# ==========================================

class GraphState(TypedDict):

    question: str

    documents: list[Document]

    web_results: list[Document]

    answer: str

    web_search_needed: bool


# ==========================================
# 9. RETRIEVAL NODE
# ==========================================

def retrieve(state: GraphState):

    print("\n--- RETRIEVING DOCUMENTS ---")

    question = state["question"]

    documents = vector_store.similarity_search(
        question,
        k=4
    )

    print(f"Retrieved {len(documents)} documents.")

    return {
        "documents": documents,
        "web_results": [],
        "web_search_needed": False
    }


# ==========================================
# 10. DOCUMENT GRADING NODE
# ==========================================

def grade_documents(state: GraphState):

    print("\n--- GRADING DOCUMENTS ---")

    question = state["question"]

    documents = state["documents"]

    relevant_documents = []

    for document in documents:

        result = document_grader.invoke(
            {
                "question": question,
                "document": document.page_content
            }
        )

        if result.binary_score == "yes":

            print("Relevant document found.")

            relevant_documents.append(document)

        else:

            print("Irrelevant document removed.")

    # Search the web if no relevant documents remain
    web_search_needed = len(relevant_documents) == 0

    print(
        f"Relevant documents: {len(relevant_documents)}"
    )

    return {
        "documents": relevant_documents,
        "web_search_needed": web_search_needed
    }


# ==========================================
# 11. CONDITIONAL ROUTING
# ==========================================

def decide_to_generate(state: GraphState):

    print("\n--- DECIDING NEXT STEP ---")

    if state["web_search_needed"]:

        print("Retrieval insufficient. Using web search.")

        return "web_search"

    print("Relevant documents available. Generating answer.")

    return "generate"


# ==========================================
# 12. WEB SEARCH NODE
# ==========================================
def web_search(state: GraphState):

    print("--- DUCKDUCKGO WEB SEARCH ---")

    question = state["question"]

    results = web_search_tool.invoke(question)

    web_documents = [
        Document(
            page_content=results,
            metadata={"source": "DuckDuckGo"}
        )
    ]

    return {
        "web_results": web_documents
    }


# ==========================================
# 13. GENERATION NODE
# ==========================================

def generate(state: GraphState):

    print("\n--- GENERATING ANSWER ---")

    question = state["question"]

    documents = state["documents"]

    web_results = state["web_results"]

    # Combine relevant vector DB documents
    # and external web search documents

    all_documents = documents + web_results

    if not all_documents:

        return {
            "answer": (
                "I could not find sufficient relevant "
                "information to answer your question."
            )
        }

    context = "\n\n".join(
        document.page_content
        for document in all_documents
    )

    response = generator.invoke(
        {
            "question": question,
            "context": context
        }
    )

    print("Answer generated successfully.")

    return {
        "answer": response.content
    }


# ==========================================
# 14. BUILD LANGGRAPH WORKFLOW
# ==========================================

workflow = StateGraph(GraphState)


# Register nodes

workflow.add_node("retrieve", retrieve)

workflow.add_node(
    "grade_documents",
    grade_documents
)

workflow.add_node(
    "web_search",
    web_search
)

workflow.add_node("generate", generate)


# Define edges

workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "grade_documents"
)


# Conditional edge

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)


# Web search to generation

workflow.add_edge(
    "web_search",
    "generate"
)


# Final node

workflow.add_edge(
    "generate",
    END
)


# Compile graph

app = workflow.compile()


# ==========================================
# 15. RUN CRAG
# ==========================================

def ask_question(question: str):

    initial_state = {
        "question": question,
        "documents": [],
        "web_results": [],
        "answer": "",
        "web_search_needed": False
    }

    result = app.invoke(initial_state)

    print("\n========== FINAL ANSWER ==========\n")

    print(result["answer"])

    return result


if __name__ == "__main__":

    while True:

        question = input(
            "\nEnter your question (or 'exit'): "
        )

        if question.lower() == "exit":
            break

        ask_question(question)

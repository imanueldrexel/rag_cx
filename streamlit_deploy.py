import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter


embedding_model = OpenAIEmbeddings(model="text-embedding-3-large") 
vector_store = InMemoryVectorStore(embedding_model).load('vector_store', embedding_model)
open_ai_llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)


# Define prompts
transform_prompt = ChatPromptTemplate.from_template("""
Given the following question, reformulate it to include both the hypothetical scenario 
and the actual query for relevant information:

Question: {question}

Instructions:
1. Identify the hypothetical elements
2. Extract the core information need
3. Combine both into a search query

Reformulated Question:
""")

response_prompt = ChatPromptTemplate.from_template("""
Instruksi:
                                                   
1. Kamu adalah asisten untuk tugas menjawab pertanyaan spesifik untuk Bank Mandiri.
2. Jika diluar konteks Bank Mandiri, katakan bahwa pertanyaannya diluar konteks. 
3. Gunakan konteks yang diberikan untuk menginformasikan alasan kamu. 
4 Reformat struktur dari jawaban, menjadi lebih lugas.
                                                              
Context: {context}

Question: {question}
""")

def create_hypothetical_rag_chain(vectorstore):

    def retrieve_docs(query: str) -> str:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = RunnableLambda(retrieve_docs)

    question_transform_chain = transform_prompt | open_ai_llm

    context_lambda = lambda x: retriever.invoke(x["question"])

    # Add logging for context
    def log_context(x):
        context = context_lambda(x)
        return context

    chain = (
        {
            "original_question": itemgetter("original_question"),
            "question": lambda x: question_transform_chain.invoke({"question": x["original_question"]}),
            "context": log_context,
        }
        | response_prompt
        | open_ai_llm
    )


    return chain


rag_chain = create_hypothetical_rag_chain(vector_store)

# Streamlit UI
st.title("Your Truly Livin Partner")

# Input field for user query
question = st.text_input("Ask Me Anything about Livin by Bank Mandiri:")

if question:
    with st.spinner("Retrieving..."):
        # Get the response from the RAG model
        response = rag_chain.invoke({"question": question,
                                     "original_question": question
                                    }
                                    )

    # Display the result
    st.markdown("Response:")
    st.markdown(response.content, unsafe_allow_html=True)  # Use this if the content is in HTML format
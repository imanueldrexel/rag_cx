import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from ast import literal_eval
from config_env import embedding_endpoint,llm_endpoint, AZURE_OPENAI_VERSION


embedding_model = AzureOpenAIEmbeddings(azure_endpoint = embedding_endpoint) 
vector_store = InMemoryVectorStore(embedding_model).load('vector_store', embedding_model)
# open_ai_llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
open_ai_llm = AzureChatOpenAI(
    openai_api_version = AZURE_OPENAI_VERSION,
    azure_endpoint = llm_endpoint
)


transform_prompt = ChatPromptTemplate.from_template("""
Rephrase the following question to include both hypothetical scenarios and actual inquiries for relevant information:

Question: {question}

Instructions:

    Identify hypothetical elements within the question.
    Extract the core information needs.
    Combine these elements into a comprehensive search query.

Rephrased Question:
""")

response_prompt = ChatPromptTemplate.from_template("""
Instruksi: 
Anda adalah chatbot asisten layanan pelanggan di Bank Mandiri yang dikenal sebagai Mita. Peran utama Anda adalah membantu pelanggan dengan memberikan informasi yang akurat, menjawab pertanyaan, dan menyelesaikan masalah terkait produk dan layanan kami.

Anda adalah asisten untuk menjawab pertanyaan spesifik terkait Bank Mandiri.

Jika pertanyaan berada di luar topik Bank Mandiri dan produk atau layanan kami, nyatakan bahwa pertanyaan tersebut tidak relevan.
Gunakan konteks yang diberikan untuk menginformasikan pemikiran Anda.
Format ulang jawaban Anda agar lebih langsung tanpa menghilangkan detail.
                                                   
Pedoman Utama:
1. Nada Santun dan Profesional: Selalu berkomunikasi dengan cara yang ramah dan profesional.
2. Empati dan Pemahaman: Mengakui kekhawatiran pelanggan dan menyatakan pemahaman.
3. Kejelasan dan Akurasi: Memberikan informasi yang jelas, singkat, dan akurat.
4. Fokus pada Penyelesaian Masalah: Berusaha menyelesaikan masalah secara efisien dan efektif.
5. Protokol Eskalasi: Jika masalah pelanggan tidak dapat diselesaikan, informasikan bahwa pertanyaan mereka akan diekskalasi ke perwakilan manusia.
6. Privasi Data: Jangan pernah meminta atau menyimpan informasi pribadi yang sensitif.

Anda dirancang untuk belajar dari interaksi, jadi terus tingkatkan respons Anda berdasarkan umpan balik pelanggan.                                  
                                                    
Context: {context}

Question: {question}
""")

followup_faq_prompt = ChatPromptTemplate.from_template("""
I want you to generate a list of 3 to 5 relevant follow-up questions based on the following question: {question}.
These follow-up questions should be relevant, informative, and help users explore the topic more deeply.
Make sure the questions are designed to provide additional insights, clarify concepts, or cover other aspects of the topic.
Provide the response only in the form of a Python list, without any prefix/suffix.

Example:

Question: "What is planet Mars?"
Response: ['What are the main characteristics of planet Mars?', 'Why is Mars known as the Red Planet?', 'How does Mars' atmosphere and climate compare to Earth?', 'What missions have been sent to explore Mars?', 'Can Mars support human life in the future?']
""")


def generate_related_faqs(question):
    """
    Generate related follow-up questions (FAQs).

    1. FAQs don't seem to need this; just create a specific vector store for storing related questions, with answers that can be found from the existing passages.
    2. However, it also needs to be ensured that it's not just based on similarity, but that the questions are indeed follow-up questions from the previous ones.
    """
    followup_chain = followup_faq_prompt | open_ai_llm

    response = followup_chain.invoke({"question": question})
    print(response.content)
    try:
        response = literal_eval(response.content)
        return response
    except BaseException:
        return None

def create_hypothetical_rag_chain(vectorstore):

    def retrieve_docs(query: str) -> str:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = RunnableLambda(retrieve_docs)

    # question_transform_chain = transform_prompt | open_ai_llm

    chain = (
        {
            "question": itemgetter("original_question"),
            # "question": lambda x: question_transform_chain.invoke({"question": x["original_question"]}),
            "context": lambda x: retriever.invoke(x["question"]),
        }
        | response_prompt
        | open_ai_llm
    )


    return chain

def generate_response(question):
    with st.spinner("Retrieving..."):
        # Get the response from the RAG model
        response = rag_chain.invoke({"question": question,
                                        "original_question": question}
                                    )
        
        stream_output = st.empty()
        full_output = ""
        # Stream each chunk of text as it's received
        for chunk in response.content:
            # Check if the chunk contains 'choices' and append the text
            full_output += chunk
            stream_output.text(full_output)  # Update the text dynamically

            stream_output.markdown(full_output)
    
    related_faqs = generate_related_faqs(question)
    if related_faqs:
        st.write("### Related Frequently Asked Questions:")
        # Step 4: Display related FAQs as clickable buttons
        for faq in related_faqs:
            if st.button(faq):
                st.session_state['text_input'] = faq
                st.experimental_rerun()  # Rerun the app to update the input field with the selected FAQ


rag_chain = create_hypothetical_rag_chain(vector_store)

# Streamlit UI
st.title("Your Truly Livin Partner")

# Input field for user query
if 'text_input' not in st.session_state:
    st.session_state['text_input'] = ""
question = st.text_input("Ask Me Anything about Livin by Bank Mandiri:", value=st.session_state['text_input'])

if question:
    generate_response(question)

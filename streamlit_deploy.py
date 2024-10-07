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
Berdasarkan pertanyaan berikut, rumuskan kembali untuk mencakup baik skenario hipotetis
maupun pertanyaan aktual untuk informasi yang relevan:

Pertanyaan: {question}

Instruksi:
1. Identifikasi elemen-elemen hipotetis
2. Ekstrak kebutuhan informasi inti
3. Gabungkan keduanya menjadi sebuah kueri pencarian

Pertanyaan yang Dirumuskan Ulang:
""")

response_prompt = ChatPromptTemplate.from_template("""
Instruksi:
                                                   
1. Kamu adalah asisten untuk tugas menjawab pertanyaan spesifik untuk Bank Mandiri
2. Jika diluar topik bank mandiri dan produk-produk serta layanannya, katakan bahwa pertanyaannya tidak relevant.
3. Gunakan konteks yang diberikan untuk menginformasikan alasan kamu. 
4. Reformat struktur dari jawaban, menjadi lebih lugas tanpa menghilangkan detail
                                                              
Context: {context}

Question: {question}
""")

followup_faq_prompt = ChatPromptTemplate.from_template("""
Saya ingin Anda menghasilkan daftar 3 hingga 5 pertanyaan lanjutan terkait berdasarkan pertanyaan berikut: {question}. 
Pertanyaan lanjutan ini harus relevan, informatif, dan membantu pengguna untuk mengeksplorasi topik lebih mendalam. 
Pastikan pertanyaannya dirancang untuk memberikan wawasan tambahan, memperjelas konsep, atau mencakup aspek lain dari topik tersebut.
Berikan response hanya bentuk python's List, jangan berikan prefix/suffix apapun.

Contoh:

Pertanyaan: "Apa itu planet Mars?"
Response: ['Apa saja ciri-ciri utama dari planet Mars?', 'Mengapa Mars dikenal sebagai Planet Merah?', 'Bagaimana perbandingan atmosfer dan iklim Mars dengan Bumi?', 'Misi apa saja yang telah dikirim untuk menjelajahi Mars?', 'Apakah Mars bisa mendukung kehidupan manusia di masa depan?']

""")


def generate_related_faqs(question):
    """
    Generate related follow-up questions (FAQs).

    1. FAQ sepertinya tidak perlu pakai ini, bikin aja 1 vector_store khusus untuk storing pertanyaan2 related, yang jawabannya bisa ditemukan dari passage yang ada.
    2. tapi perlu di make sure juga kalau jangan hanya similarity, tapi memang follow up question dari pertanyaan sebelumnya.
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

    question_transform_chain = transform_prompt | open_ai_llm

    chain = (
        {
            "original_question": itemgetter("original_question"),
            "question": lambda x: question_transform_chain.invoke({"question": x["original_question"]}),
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

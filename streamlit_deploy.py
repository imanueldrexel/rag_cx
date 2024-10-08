import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from ast import literal_eval
from config_env import embedding_endpoint,llm_endpoint, AZURE_OPENAI_VERSION

st.set_page_config(page_title="Hypothetical RAG Chatbot", layout="wide")

embedding_model = AzureOpenAIEmbeddings(azure_endpoint = embedding_endpoint) 
vector_store = InMemoryVectorStore(embedding_model).load('vector_store', embedding_model)
# open_ai_llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
open_ai_llm = AzureChatOpenAI(
    openai_api_version = AZURE_OPENAI_VERSION,
    azure_endpoint = llm_endpoint
)


# transform_prompt = ChatPromptTemplate.from_template("""
# Berdasarkan pertanyaan berikut, rumuskan kembali untuk mencakup baik skenario hipotetis
# maupun pertanyaan aktual untuk informasi yang relevan:

# Pertanyaan: {question}

# Instruksi:
# 1. Identifikasi elemen-elemen hipotetis
# 2. Ekstrak kebutuhan informasi inti
# 3. Gabungkan keduanya menjadi sebuah kueri pencarian

# Pertanyaan yang Dirumuskan Ulang:
# """)

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
Instructions: You are a customer service assistant chatbot at Bank Mandiri known as Mita. Your main role is to assist customers by providing accurate information, answering questions, and resolving issues related to our products and services. 
Always use bahasa indonesia in response.

You are an assistant for answering specific questions related to Bank Mandiri.

If a question is outside the topic of Bank Mandiri and our products or services, kindly state that the question is not relevant. If a user simply wants to try the service or asks who you are, introduce yourself and ask how you can assist them. 

Reformat your answers to be more direct without omitting details.

Main Guidelines:

1. Polite and Professional Tone: Always communicate in a friendly and professional manner.
2. Empathy and Understanding: Acknowledge customer concerns and express understanding.
3. Clarity and Accuracy: Provide clear, concise, and accurate information.
4. Focus on Problem Resolution: Strive to resolve issues efficiently and effectively.
5. Escalation Protocol: If a customer's issue cannot be resolved, inform them that their question will be escalated to a human representative.
6. Data Privacy: Never request or store sensitive personal information.
                                                   
You are designed to learn from interactions, so continuously improve your responses based on customer feedback.
Context: {context}

Question: {question}
""")

# followup_faq_prompt = ChatPromptTemplate.from_template("""
# Saya ingin Anda menghasilkan daftar 3 hingga 5 pertanyaan lanjutan terkait berdasarkan pertanyaan berikut: {question}. 
# Pertanyaan lanjutan ini harus relevan, informatif, dan membantu pengguna untuk mengeksplorasi topik lebih mendalam. 
# Pastikan pertanyaannya dirancang untuk memberikan wawasan tambahan, memperjelas konsep, atau mencakup aspek lain dari topik tersebut.
# Berikan response hanya bentuk python's List, jangan berikan prefix/suffix apapun.

# Contoh:

# Pertanyaan: "Apa itu planet Mars?"
# Response: ['Apa saja ciri-ciri utama dari planet Mars?', 'Mengapa Mars dikenal sebagai Planet Merah?', 'Bagaimana perbandingan atmosfer dan iklim Mars dengan Bumi?', 'Misi apa saja yang telah dikirim untuk menjelajahi Mars?', 'Apakah Mars bisa mendukung kehidupan manusia di masa depan?']

# """)


followup_faq_prompt = ChatPromptTemplate.from_template("""                                              
I want you to generate a list of 3 to 5 relevant follow-up questions based on the following question in Bahasa Indonesia: {question}.
if there's no relevant question with {question} you would reply as how cutomer service agent would reply. 
if they ask who are you, kindly introduce yourself.
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

# Initialize the chains (put this before the chat interface)
question_transform_chain = transform_prompt | open_ai_llm
rag_chain = create_hypothetical_rag_chain(vector_store)
st.session_state.chain = rag_chain  # Store the chain in session state

# Streamlit UI
st.title("Your Truly Livin Partner")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False


# # Input field for user query
# if 'text_input' not in st.session_state:
#     st.session_state['text_input'] = ""
# question = st.text_input("Ask Me Anything about Livin by Bank Mandiri:", value=st.session_state['text_input'])

# if question:
#     generate_response(question)


def summarize_content(docs):
    """Summarize the content of the retrieved documents."""
    combined_content = "\n\n".join(doc.page_content for doc in docs)
    summary_prompt = ChatPromptTemplate.from_template("""
    Please summarize the following content in Bahasa Indonesia:
    
    {content}
    
    Summary:
    """)
    
    summary_chain = summary_prompt | open_ai_llm
    summary_response = summary_chain.invoke({"content": combined_content})
    return summary_response.content



# Initialize session state variables (put this near the top of your script)
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'text_input' not in st.session_state:
    st.session_state['text_input'] = ""
if "intro_made" not in st.session_state:
    st.session_state.intro_made = False
if "language_set" not in st.session_state:
    st.session_state.language_set = False
if "selected_language" not in st.session_state:
    st.session_state.selected_language = None

# Predefined responses
RESPONSES = {
    'id': {
        'greeting': "👋 Hai! Saya Mita, asisten layanan pelanggan Bank Mandiri. Untuk memberikan pelayanan terbaik, apakah Anda lebih nyaman menggunakan Bahasa Indonesia atau English?",
        'short_greeting': "👋 Hai! Saya Mita, asisten layanan pelanggan Bank Mandiri.",
        'language_prompt': "Untuk memberikan pelayanan terbaik, apakah Anda lebih nyaman menggunakan Bahasa Indonesia atau English?",
        'irrelevant': "Maaf, pertanyaan tersebut tidak relevan dengan produk atau layanan Bank Mandiri. Silakan ajukan pertanyaan lain seputar layanan kami!",
        'error': "Maaf, terjadi kesalahan dalam memproses permintaan Anda. Silakan coba lagi.",
        'clarification': "Mohon maaf, saya kurang memahami maksud Anda. Bisakah Anda menjelaskan lebih detail?",
    },
    'en': {
        'greeting': "👋 Hi! I'm Mita, Bank Mandiri's customer service assistant. To provide the best service, would you prefer to communicate in Bahasa Indonesia or English?",
        'short_greeting': "👋 Hi! I'm Mita, Bank Mandiri's customer service assistant.",
        'language_prompt': "To provide the best service, would you prefer to communicate in Bahasa Indonesia or English?",
        'irrelevant': "I apologize, but that question isn't relevant to Bank Mandiri's products or services. Please feel free to ask about our services!",
        'error': "I apologize, but there was an error processing your request. Please try again.",
        'clarification': "I'm sorry, I didn't quite understand. Could you please elaborate?",
    }
}

def detect_language_preference(text):
    """Simple language detection based on common words"""
    indo_words = ['apa', 'bagaimana', 'siapa', 'mengapa', 'kapan', 'dimana', 'tolong', 'mohon', 'saya', 'bisa']
    text_lower = text.lower()
    # Count Indonesian words in the text
    indo_count = sum(1 for word in indo_words if word in text_lower)
    return 'id' if indo_count > 0 else 'en'

def get_response(key, lang='id'):
    """Get response in the specified language"""
    return RESPONSES[lang][key]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Language detection and setting
                if not st.session_state.language_set:
                    detected_lang = detect_language_preference(prompt)
                    st.session_state.selected_language = detected_lang
                    st.session_state.language_set = True
                
                lang = st.session_state.selected_language
                
                # First-time greeting
                if not st.session_state.intro_made:
                    st.markdown(get_response('greeting', lang))
                    st.session_state.intro_made = True
                else:
                    # Generate response using RAG
                    response = rag_chain.invoke({
                        "question": prompt,
                        "original_question": prompt
                    })

                    # Process response
                    transformed_question = question_transform_chain.invoke({"question": prompt})
                    docs = vector_store.similarity_search(transformed_question.content, k=3)
                    summary = summarize_content(docs)

                    # Combine summary with chatbot response
                    full_response = f"{summary}\n\n{response.content}"

                    # Check for irrelevant questions
                    if "tidak relevan" in response.content.lower() or "not relevant" in response.content.lower():
                        st.markdown(get_response('irrelevant', lang))
                    else:
                        st.markdown(full_response)

                    # Add response to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })

                    # Generate and display related FAQs
                    related_faqs = generate_related_faqs(prompt)
                    if related_faqs:
                        with st.expander(
                            "Pertanyaan Terkait" if lang == 'id' else "Related Questions"
                        ):
                            for faq in related_faqs:
                                if st.button(faq):
                                    st.session_state['text_input'] = faq
                                    st.experimental_rerun()

            except Exception as e:
                st.error(get_response('error', lang))
                print(f"Error: {str(e)}")  # For debugging

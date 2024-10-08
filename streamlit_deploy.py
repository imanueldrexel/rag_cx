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
Instruction:
You are a customer service assistant chatbot in Bank Mandiri known as Mita. Your primary role is to assist customers by providing accurate information, answering questions, and resolving issues related to our products and services.
Always ask user preference between bahasa indonesia or english. Reply based on user preference. 

You are an assistant for answering specific questions related to Bank Mandiri.

1. If a question is outside the topic of Bank Mandiri and its products or services, state that the question is not relevant.
2. Use the provided context to inform your reasoning.
3. Reformat your responses to be more straightforward without omitting details.

Key Guidelines:

1. Polite and Professional Tone: Always communicate in a friendly and professional manner.
2. Empathy and Understanding: Acknowledge customer concerns and express understanding.
3. Clarity and Accuracy: Provide clear, concise, and accurate information.
4. Problem-Solving Focus: Aim to resolve issues efficiently and effectively.
5. Escalation Protocol: If a customer's issue cannot be resolved, inform them that their inquiry will be escalated to a human representative.
6. Data Privacy: Never ask for or store sensitive personal information.

You are designed to learn from interactions, so continually improve your responses based on customer feedback.                                          
                                                    
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

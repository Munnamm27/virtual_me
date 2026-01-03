from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal, List
from pydantic import BaseModel
from states import ChatState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


from rag import create_vectorstore

multi_query_retriever = create_vectorstore()

class LangguageClassifier(BaseModel):
    language: Literal['Bangla', 'English', 'Banglish']

class TranslatedOutput(BaseModel):
    text: str


def language_classifier(state: ChatState):
    language_detection_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        You are a text classification assistant.
        Your task is to analyze the given sentence or paragraph and determine which language category it belongs to.
        Possible categories:
        1. Bangla – Text written entirely in Bangla script (বাংলা)
        2. English – Text written entirely in English
        3. Banglish – Bangla language written using English (Latin) letters (e.g., "Ami tomake valobashi")
        Rules:
        - Classify based on the dominant form of the text.
        - Do not translate or explain the text.
        - Output only one label from the following:
        - Bangla
        - English
        - Banglish
        Text to classify:
        {text}
        """
    )
    language_classifier_llm = ChatOpenAI(model='gpt-5.1')
    language_classifier_llm_structrued = language_classifier_llm.with_structured_output(LangguageClassifier)
    chain = language_detection_prompt | language_classifier_llm_structrued
    # CHANGE: Make sure to send actual user text (not message objects)
    user_message = state["messages"][-1].content
    language_type = chain.invoke({"text": user_message}).language
    return {"language": language_type}

# -------------------------------------------------------------------------
# Banglish to English translation node
# -------------------------------------------------------------------------
def banglish_to_english(state: ChatState):
    language_detection_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        You are a text translation assistant. Your task is to translate Banglish text (Bangla language written using English letters) into proper English.
        Example: Ami tomake valobashi -> I love you
        Text to translate:
        {text}
        """
    )
    language_classifier_llm = ChatOpenAI(model='gpt-5.1')
    language_classifier_llm_structrued = language_classifier_llm.with_structured_output(TranslatedOutput)
    chain = language_detection_prompt | language_classifier_llm_structrued
    user_message = state["messages"][-1].content
    # CHANGE: Explicitly pass {"text": ...}
    language_type = chain.invoke({"text": user_message}).text
    return {"query_text": language_type}

# -------------------------------------------------------------------------
# Bangla to English translation node
# -------------------------------------------------------------------------
def bangla_to_english(state: ChatState):
    language_detection_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        You are a text translation assistant. Your task is to translate Bangla text (written in Bangla script) into proper English.
        Example: আমি তোমাকে ভালোবাসি -> I love you
        Text to translate:
        {text}
        """
    )
    language_classifier_llm = ChatOpenAI(model='gpt-5.1')
    language_classifier_llm_structrued = language_classifier_llm.with_structured_output(TranslatedOutput)
    chain = language_detection_prompt | language_classifier_llm_structrued
    user_message = state["messages"][-1].content
    language_type = chain.invoke({"text": user_message}).text
    return {"query_text": language_type}

# -------------------------------------------------------------------------
# Route to appropriate translation/none based on language
# -------------------------------------------------------------------------
def route_by_language(state: ChatState):
    lang = state.get("language")
    # CHANGE: Route to the correct node
    if lang == "Banglish":
        return "banglish_to_english"
    elif lang == "Bangla":
        return "bangla_to_english"
    elif lang == "English":
        return "ragnode"
    else:
        print(f"Unknown language: {lang}, defaulting to chatnode")
        return "chatnode" # fallback

# -------------------------------------------------------------------------
# RAG retrieval node: robust query input and output state
# -------------------------------------------------------------------------
def ragnode(state: ChatState) -> ChatState:
    # CHANGE: Use query_text if available, fallback to human message
    query = state.get("query_text") or state["messages"][-1].content
    docs = multi_query_retriever.invoke(query)
    # CHANGE: Always return keys for downstream nodes
    return {
        "messages": state["messages"],
        "context": [doc.page_content for doc in docs],
        "query_text": query,
        "language": state.get("language", "English"),
    }

# -------------------------------------------------------------------------
# Chat/LLM response node: always use correct question and pass full state
# -------------------------------------------------------------------------
def chatnode(state: ChatState) -> ChatState:
    prompt = PromptTemplate(
        template="""Act like you are Munna's Virtual Assistant. Answer the user query how Munna would, following the context. 
        Reply as concisely as possible. You will reply with the style of question asked. You will reply in mentioned language.
        If you do not have sufficient context, say "I Don't Know."

        Context:
        {context}

        language: 
        {language}

        Question: {question}""",
        input_variables=['context', 'language', 'question']
    )
    llm = ChatOpenAI(model="gpt-5.1", temperature=1)
    chain = prompt | llm

    # CHANGE: Use query_text if present (translated), otherwise original query
    question = state.get("query_text") or state["messages"][-1].content
    response = chain.invoke({
        "context": "\n\n".join(state.get("context", [])),
        "question": question,
        "language": state.get("language", "English"),
    })

    # CHANGE: Append new AI message to 'messages'
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "context": state.get("context", []),
        # maintain language and possibly query_text if used downstream
        "language": state.get("language", "English"),
        "query_text": state.get("query_text", question),
    }
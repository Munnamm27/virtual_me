from typing import TypedDict, Annotated, Literal,List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ChatState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    language: Literal['Bangla', 'English', 'Banglish']
    query_text: str
    context: List[str]
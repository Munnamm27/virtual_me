from langgraph.graph import StateGraph, START, END
from nodes import *
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
# -------------------------------------------------------------------------
# Workflow graph setup
# -------------------------------------------------------------------------
workflow = StateGraph(ChatState)
workflow.add_node('language_classifier', language_classifier)
workflow.add_node("banglish_to_english", banglish_to_english)
workflow.add_node("bangla_to_english", bangla_to_english)
workflow.add_node("ragnode", ragnode)
workflow.add_node("chatnode", chatnode)

workflow.add_edge(START, "language_classifier")
workflow.add_conditional_edges(
    "language_classifier",
    route_by_language
)
workflow.add_edge("bangla_to_english", "ragnode")
workflow.add_edge("banglish_to_english", "ragnode")
workflow.add_edge("ragnode", "chatnode")
workflow.add_edge("chatnode", END)

chat = workflow.compile(checkpointer=MemorySaver())

while True:
     user_input = input("You: ")
     if user_input.lower() in ["exit", "quit"]:
         break
     result = chat.invoke(
         {
             "messages": [HumanMessage(content=user_input)],
         },
         config={"configurable": {"thread_id": "chat-102003"}},
     )
     print("Bot:", result["messages"][-1].content)

# ------------------------------------------------------------------------
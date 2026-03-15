import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message is emotional or logical"
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None


def classify_message(state: State):

    last_message = state["messages"][-1]

    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - emotional
            - logical"""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])

    return {
        "message_type": result.message_type,
        "next": result.message_type
    }


def router(state: State):
    return {"next": state.get("message_type", "logical")}


def therapist_agent(state: State):

    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a compassionate therapist.
            Show empathy and help process emotions."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    reply = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):

    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a logical assistant.
            Focus only on facts and clear answers."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    reply = llm.invoke(messages)

    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "emotional": "therapist",
        "logical": "logical"
    }
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()


def run_chatbot():

    state = {"messages": [], "message_type": None}

    while True:

        user_input = input("Message: ")

        if user_input.lower() == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            HumanMessage(content=user_input)
        ]

        state = graph.invoke(state)

        if state.get("messages"):
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()
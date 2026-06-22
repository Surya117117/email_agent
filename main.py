from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from classifier import classify_email
from priority import get_priority

class EmailState(TypedDict):
    sender: str
    subject: str
    body: str
    category: str
    priority: str



def classify_node(state: EmailState):
    category = classify_email(
        state["subject"] + state["body"]
    )
    return {
        "category": category
    }

builder = StateGraph(EmailState)

builder.add_node("classifier",classify_node)
builder.add_node("priority", get_priority)

builder.add_edge(START, "classifier")
builder.add_edge("classifier","priority")
builder.add_edge("priority", END)

graph = builder.compile()

result = graph.invoke(
    {
        "sender" : "owner@company.com",
        "subject" : """This email is related to leetcode problem solving""",
        "body":
        """Hey there, One thing that helped me a lot when practicing LeetCode was making sure I practiced like I play. And honestly, this is one of the biggest mistakes people make with coding interview prep. You can do as many problems as you want. But if you don't emulate a real coding interview, you're going to struggle when you're actually thrown into that environment.
        """
    }
)
print(result)

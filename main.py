from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from classifier import classify_email

class EmailState(TypedDict):
    email: str
    category : str

def classify_node(state: EmailState):
    category = classify_email(
        state["email"]
    )
    return {
        "category": category
    }

builder = StateGraph(EmailState)

builder.add_node("classifier",classify_node)
builder.add_edge(START, "classifier")
builder.add_edge("classifier", END)

graph = builder.compile()

result = graph.invoke(
    {
        "email":
        """Hey there, One thing that helped me a lot when practicing LeetCode was making sure I practiced like I play. And honestly, this is one of the biggest mistakes people make with coding interview prep. You can do as many problems as you want. But if you don't emulate a real coding interview, you're going to struggle when you're actually thrown into that environment.
        """
    }
)
print(result)

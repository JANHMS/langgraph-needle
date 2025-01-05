from typing import Annotated, List, Union, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import Optional
from needle_tools import create_collection, add_files_to_collection, check_indexing_status

# Initialize the model with tools
model = ChatOpenAI(model="gpt-4", temperature=0).bind_tools([
    create_collection, 
    add_files_to_collection, 
    check_indexing_status
])

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    
    if not messages:
        return {"messages": [HumanMessage(content="""Let's do the following:
1. Create a new collection named 'Documentation Collection'
2. Add these documentation URLs to it:
   - https://docs.needle-ai.com
   - https://docs.needle-ai.com/docs/langchain/""")]}
    
    # Check the last message
    last_message = messages[-1]
    if isinstance(last_message, ToolMessage) and last_message.content == "Files added successfully":
        return {"messages": messages}  # Don't add any new messages
        
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the function that determines whether to continue
class NeedleState(MessagesState, TypedDict):
    collection_id: Optional[str]
    files_added: bool

def should_continue(state: MessagesState) -> Literal["tools", "agent", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    print(f"Debug - Message type: {type(last_message)}")  # Debug print
    print(f"Debug - Message content: {last_message.content}")  # Debug print
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # End immediately after files are added
    if isinstance(last_message, ToolMessage):
        print(f"Debug - Tool message received: {last_message.content}")  # Debug print
        if last_message.content == "Files added successfully":
            print("Debug - Ending workflow after files added")  # Debug print
            return END
    
    # Otherwise continue with agent
    return "agent"

# Initialize state properly
initial_state = {
    "messages": [],
    "collection_id": None,
    "files_added": False
}

# Create and compile the graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([create_collection, add_files_to_collection, check_indexing_status]))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "agent": "agent", END: END}
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Run the graph
initial_state = {"messages": []}

try:
    print("\nStarting Needle operations...")
    for step in app.stream(initial_state):
        if "messages" in step:
            for msg in step["messages"]:
                msg.pretty_print()

except Exception as e:
    print(f"\nError during execution: {str(e)}")
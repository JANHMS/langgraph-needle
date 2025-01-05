import os
from typing import Annotated, List, Union, TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers.needle import NeedleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.needle import NeedleLoader


# Get environment variables
NEEDLE_API_KEY = os.environ.get("NEEDLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not NEEDLE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Required environment variables NEEDLE_API_KEY and OPENAI_API_KEY must be set")

# Initialize Needle with existing collection
COLLECTION_ID = "clt_01JGS1C3PFZCY18B0948D9WT60"

# Define our state
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], add_messages]
    file_added: bool
    indexed: bool

# Initialize Needle components
needle_loader = NeedleLoader(
    needle_api_key=NEEDLE_API_KEY,
    collection_id=COLLECTION_ID
)

retriever = NeedleRetriever(
    needle_api_key=NEEDLE_API_KEY,
    collection_id=COLLECTION_ID
)

# Define tools for Needle operations
@tool
def add_file_to_collection() -> str:
    """Add the Tech Radar file to the collection."""
    files = {
        "docs.needle-ai.com": "https://docs.needle-ai.com"
    }
    needle_loader.add_files(files=files)
    return "File added successfully"

@tool
def search_collection(query: str) -> str:
    """Search the collection with the given query."""
    # Create RAG chain
    llm = ChatOpenAI(temperature=0)
    
    # Define system prompt
    system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know, say so concisely.\n\n{context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Create question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Execute search
    response = rag_chain.invoke({"input": query})
    return str(response['answer']) if 'answer' in response else str(response)

# Initialize the model with tools
model = ChatOpenAI(model="gpt-4", temperature=0).bind_tools([add_file_to_collection, search_collection])

def should_continue(state: AgentState) -> Literal["tools", "agent", END]:
    messages = state['messages']
    
    if not messages:
        return "agent"

    last_message = messages[-1]
    
    # Handle tool responses
    if isinstance(last_message, ToolMessage):
        if last_message.content == "File added successfully":
            state['file_added'] = True
            return "agent"
        # End immediately after getting any search result
        return END

    # Handle AI messages
    if isinstance(last_message, AIMessage):
        # If we already have a search result in our history, end
        if any(isinstance(m, ToolMessage) and m.content != "File added successfully" for m in messages[:-1]):
            return END
        # If we have tool calls and haven't indexed yet
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if state.get('file_added') and not state.get('indexed'):
                return "agent"  # Wait for indexing before executing search
            return "tools"

    return "agent"

def call_model(state: AgentState):
    messages = state['messages']
    
    if not messages:
        return {"messages": [HumanMessage(content="Let's add the file to our Needle collection and then search for what it says about Needle.")]}
    
    # If file was just added, wait 30 seconds before allowing search
    if state.get('file_added') and not state.get('indexed'):
        print("\nWaiting 30 seconds for file to be indexed...")
        import time
        time.sleep(30)
        state['indexed'] = True
        return {"messages": [HumanMessage(content="File has been indexed. Now searching for information about Needle...")]}
    
    response = model.invoke(messages)
    return {"messages": [response]}

# Create and compile the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([add_file_to_collection, search_collection]))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "agent": "agent", END: END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Run the graph
initial_state = {
    "messages": [],
    "file_added": False,
    "indexed": False
}


print("\nStarting Needle operations...")
for step in app.stream(initial_state, {"recursion_limit": 10}):
    if "messages" in step.get("agent", {}):
        for msg in step["agent"]["messages"]:
            msg.pretty_print()
    elif "messages" in step.get("tools", {}):
        for msg in step["tools"]["messages"]:
            msg.pretty_print()

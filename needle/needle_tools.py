from langchain_core.tools import tool
from needle.v1 import NeedleClient
from needle.v1.models import FileToAdd
from typing import Optional, List

# Initialize Needle client
ndl = NeedleClient()

# Collection Management Tools
@tool
def create_collection(name: str) -> str:
    """Create a new Needle collection with a specified name.
    
    Args:
        name (str): The name to give to the new collection
    """
    collection = ndl.collections.create(name=name)
    print(f"Created collection: {collection.id} with name: {name}")
    return collection.id

@tool
def list_collections() -> str:
    """List all available collections."""
    collections = ndl.collections.list()
    collection_info = [f"ID: {c.id}, Name: {c.name}" for c in collections]
    print("\n".join(collection_info))
    return str(collection_info)

@tool
def get_collection(collection_id: str) -> str:
    """Get details of a specific collection."""
    collection = ndl.collections.get(collection_id)
    print(f"Collection details: {collection.name} (ID: {collection.id})")
    return str(collection.__dict__)

@tool
def get_collection_stats(collection_id: str) -> str:
    """Get statistics for a collection."""
    stats = ndl.collections.get_stats(collection_id)
    print(f"Collection stats: {stats}")
    return str(stats.__dict__)

# File Management Tools
@tool
def add_files_to_collection(collection_id: str, urls: List[str]) -> str:
    """Add files to the collection from a list of URLs.
    
    Args:
        collection_id (str): The ID of the collection to add files to
        urls (List[str]): List of URLs to add to the collection
    """
    files = ndl.collections.files.add(
        collection_id=collection_id,
        files=[
            FileToAdd(
                name=f"doc_{i+1}",  # Generate a simple name based on index
                url=url
            )
            for i, url in enumerate(urls)
        ]
    )
    print(f"Added files: {[f.name for f in files]}")
    return "Files added successfully"

@tool
def list_collection_files(collection_id: str) -> str:
    """List all files in a collection."""
    files = ndl.collections.files.list(collection_id)
    file_info = [f"Name: {f.name}, Status: {f.status}" for f in files]
    print("\n".join(file_info))
    return str(file_info)

@tool
def get_file_download_url(file_id: str) -> str:
    """Get a download URL for a specific file."""
    url = ndl.files.get_download_url(file_id)
    print(f"Download URL: {url}")
    return url

# Search Tools
@tool
def search_collection(collection_id: str, query: str, top_k: Optional[int] = 5) -> str:
    """Search within a collection using semantic search."""
    results = ndl.collections.search(
        collection_id=collection_id,
        text=query,
        top_k=top_k
    )
    print(f"Found {len(results)} results")
    return str([r.__dict__ for r in results])

@tool
def check_indexing_status(collection_id: str) -> str:
    """Check if all files in the collection are indexed."""
    files = ndl.collections.files.list(collection_id)
    all_indexed = all(f.status == "indexed" for f in files)
    print(f"Indexing status: {'Complete' if all_indexed else 'In Progress'}")
    return str(all_indexed) 
# Needle LangGraph Integration

This directory contains examples of integrating Needle with LangGraph for creating conversational agents that can manage document collections and perform semantic search.

## Files

- `needle_tools.py`: Contains all the Needle-specific tools wrapped for LangGraph usage
- `needle_with_client.py`: Example implementation of a LangGraph agent that creates collections and adds documents

## Available Tools

### Collection Management
- `create_collection(name: str)`: Creates a new collection with specified name
- `list_collections()`: Lists all available collections
- `get_collection(collection_id: str)`: Gets details of a specific collection
- `get_collection_stats(collection_id: str)`: Gets statistics for a collection

### File Management
- `add_files_to_collection(collection_id: str, urls: List[str])`: Adds files from URLs to a collection
- `list_collection_files(collection_id: str)`: Lists all files in a collection
- `get_file_download_url(file_id: str)`: Gets a download URL for a specific file

### Search
- `search_collection(collection_id: str, query: str, top_k: Optional[int])`: Performs semantic search
- `check_indexing_status(collection_id: str)`: Checks if files are indexed

## Example Usage

The example in `needle_with_client.py` demonstrates:
1. Creating a new collection
2. Adding documentation URLs to the collection
3. Checking indexing status
4. Automatic cleanup of resources

To run the example:
```bash
python needle_with_client.py
```

Make sure you have set up your Needle API credentials before running the examples. 
`EXPORT NEEDLE_API_KEY=234`
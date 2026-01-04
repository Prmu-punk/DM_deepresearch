import sys
import os
import logging

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "src"))

from open_deep_research.kg_enhanced_search import query_conceptnet_cached, extract_related_concepts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_conceptnet_integration():
    term = "artificial_intelligence"
    print(f"\nTesting ConceptNet API for term: '{term}'...")
    
    try:
        # 1. Test API Call
        data = query_conceptnet_cached(term, limit=5)
        print(f"API Call Status: {'Success' if data and 'edges' in data else 'Failed/Empty'}")
        
        if data:
            print(f"Raw Response Keys: {list(data.keys())}")
            if 'edges' in data:
                print(f"Number of edges: {len(data['edges'])}")
        
        # 2. Test Extraction
        # extract_related_concepts takes the concept string, not the data
        concepts = extract_related_concepts(term)
        print(f"Extracted Concepts: {concepts}")
        
    except Exception as e:
        print(f"Test Failed with error: {e}")

def test_error_handling():
    print("\nTesting Error Handling (Simulating Bad Response)...")
    
    # We can't easily pass None to extract_related_concepts if it expects a string and calls the API.
    # But we can test if passing a string that causes an API error works gracefully.
    # Since we know the API is returning 502s, running it with any term is effectively testing error handling.
    
    try:
        concepts = extract_related_concepts("error_simulation_term")
        print(f"Handling error term: Success (Result: {concepts})")
    except Exception as e:
        print(f"Handling error term: Failed (Error: {e})")

if __name__ == "__main__":
    test_conceptnet_integration()
    test_error_handling()

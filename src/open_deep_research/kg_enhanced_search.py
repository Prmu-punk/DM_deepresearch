"""Knowledge Graph Enhanced Search using ConceptNet and Tavily.

This module provides functionality to enhance Tavily web searches using
ConceptNet knowledge graph for query expansion and concept enrichment.
"""

import asyncio
import logging
import requests
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

# ConceptNet API configuration
CONCEPTNET_API_BASE = "http://api.conceptnet.io"
CONCEPTNET_TIMEOUT = 5  # seconds

# Logger setup
logger = logging.getLogger(__name__)


##########################
# ConceptNet Query Utils
##########################

@lru_cache(maxsize=1000)
def query_conceptnet_cached(concept: str, limit: int = 20) -> Dict[str, Any]:
    """Query ConceptNet API with caching to reduce API calls.
    
    Args:
        concept: The concept to query (e.g., "climate_change")
        limit: Maximum number of edges to return
        
    Returns:
        JSON response from ConceptNet API
    """
    # Clean and format the concept for API
    concept_clean = concept.lower().strip().replace(" ", "_")
    url = f"{CONCEPTNET_API_BASE}/c/en/{concept_clean}"
    params = {"limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=CONCEPTNET_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"ConceptNet API returned status {response.status_code} for '{concept}'")
    except requests.Timeout:
        logger.warning(f"ConceptNet API timeout for '{concept}'")
    except requests.RequestException as e:
        logger.warning(f"ConceptNet API error for '{concept}': {e}")
    
        return {"edges": []}


def extract_related_concepts(
    concept: str,
    relation_types: Optional[List[str]] = None,
    max_concepts: int = 10
) -> List[Dict[str, str]]:
    """Extract related concepts from ConceptNet.
    
    Args:
        concept: The source concept to find relations for
        relation_types: List of relation types to include (None = all useful types)
        max_concepts: Maximum number of related concepts to return
        
    Returns:
        List of dicts with 'concept' and 'relation' keys
    """
    # Default relation types useful for search enhancement
    if relation_types is None:
        relation_types = [
            "RelatedTo",      # General relatedness
            "Synonym",        # Same meaning
            "SimilarTo",      # Similar concepts
            "IsA",            # Hypernym (is a type of)
            "PartOf",         # Meronym (part of)
            "HasA",           # Has component
            "UsedFor",        # Purpose/usage
            "Causes",         # Causal relationship
            "HasProperty",    # Properties
            "DefinedAs",      # Definitions
        ]
    
    data = query_conceptnet_cached(concept)
    related = []
    seen_concepts = set()
    concept_lower = concept.lower()
    
    for edge in data.get("edges", []):
        # Extract relation type
        rel_label = edge.get("rel", {}).get("label", "")
        
        # Filter by relation type
        if rel_label not in relation_types:
            continue
        
        # Extract start and end concepts
        start_info = edge.get("start", {})
        end_info = edge.get("end", {})
        
        start_label = start_info.get("label", "")
        end_label = end_info.get("label", "")
        start_lang = start_info.get("language", "en")
        end_lang = end_info.get("language", "en")
        
        # Only include English concepts
        if start_lang == "en" and start_label.lower() != concept_lower:
            if start_label.lower() not in seen_concepts:
                seen_concepts.add(start_label.lower())
                related.append({
                    "concept": start_label,
                    "relation": rel_label,
                    "weight": edge.get("weight", 1.0)
                })
        
        if end_lang == "en" and end_label.lower() != concept_lower:
            if end_label.lower() not in seen_concepts:
                seen_concepts.add(end_label.lower())
                related.append({
                    "concept": end_label,
                    "relation": rel_label,
                    "weight": edge.get("weight", 1.0)
                })
        
        if len(related) >= max_concepts:
            break
    
    # Sort by weight (higher = more relevant)
    related.sort(key=lambda x: x.get("weight", 0), reverse=True)
    
    return related[:max_concepts]


def get_synonyms(concept: str, max_count: int = 5) -> List[str]:
    """Get synonyms for a concept from ConceptNet.
    
    Args:
        concept: The concept to find synonyms for
        max_count: Maximum number of synonyms to return
        
    Returns:
        List of synonym strings
    """
    related = extract_related_concepts(
        concept, 
        relation_types=["Synonym", "SimilarTo"],
        max_concepts=max_count
    )
    return [r["concept"] for r in related]


##########################
# Enhancement Decision Logic
##########################

def should_enhance_with_kg(
    query: str,
    force_enhancement: bool = False,
    max_word_count: int = 5
) -> bool:
    """Determine if a query should be enhanced using knowledge graph.
    
    Simple rule: enhance if word count < max_word_count.
    
    Args:
        query: The search query to evaluate
        force_enhancement: If True, always return True
        max_word_count: Maximum words for enhancement (default: 5)
        
    Returns:
        True if query should be enhanced with knowledge graph
    """
    if force_enhancement:
        return True
    
    # Simple rule: word count < 5 -> enhance
    words = query.strip().split()
    word_count = len(words)
    
    return word_count < max_word_count


##########################
# Query Enhancement Functions
##########################

def expand_query_with_kg(
    query: str,
    max_expansion_terms: int = 3,
    expansion_strategy: Literal["append", "or", "separate"] = "or"
) -> List[str]:
    """Expand a search query using ConceptNet knowledge graph.
    
    Args:
        query: Original search query
        max_expansion_terms: Maximum number of expansion terms to add
        expansion_strategy: How to combine expansions
            - "append": Add terms to original query
            - "or": Create OR query with related terms
            - "separate": Return separate queries for each expansion
            
    Returns:
        List of expanded query strings
    """
    # Extract main keywords (words > 3 chars, not stopwords)
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "what", "how", "why"}
    keywords = [w for w in query.split() if len(w) > 3 and w.lower() not in stopwords]
    
    if not keywords:
        return [query]
    
    # Get related concepts for main keywords
    all_related = []
    for keyword in keywords[:3]:  # Process top 2 keywords
        related = extract_related_concepts(keyword, max_concepts=5)
        all_related.extend(related)
    
    # Deduplicate and sort by weight
    seen = set()
    unique_related = []
    for r in all_related:
        concept_lower = r["concept"].lower()
        if concept_lower not in seen and concept_lower not in query.lower():
            seen.add(concept_lower)
            unique_related.append(r)
    
    unique_related.sort(key=lambda x: x.get("weight", 0), reverse=True)
    expansion_terms = [r["concept"] for r in unique_related[:max_expansion_terms]]
    
    if not expansion_terms:
        return [query]
    
    # Apply expansion strategy
    if expansion_strategy == "append":
        # Add terms to the end of the query
        expanded = f"{query} {' '.join(expansion_terms)}"
        return [expanded]
    
    elif expansion_strategy == "or":
        # Create OR query
        or_terms = " OR ".join([f'"{t}"' for t in expansion_terms])
        expanded = f"{query} ({or_terms})"
        return [expanded]
    
    elif expansion_strategy == "separate":
        # Return original + separate queries for each term
        queries = [query]
        for term in expansion_terms:
            queries.append(f"{query} {term}")
        return queries
    
    return [query]


def build_kg_context(query: str, max_concepts: int = 10) -> str:
    """Build a knowledge graph context summary for a query.
    
    This provides additional context that can be used by the LLM
    to better understand the query domain.
    
    Args:
        query: The search query or research topic
        max_concepts: Maximum related concepts to include per keyword
        
    Returns:
        Formatted string with KG context information
    """
    # Extract keywords (words > 3 chars, not stopwords)
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "what", "how", "why", "are", "is"}
    keywords = [w for w in query.split() if len(w) > 3 and w.lower() not in stopwords]
    
    if not keywords:
        return ""
    
    context_parts = []
    
    for keyword in keywords[:5]:  # Process top 5 keywords
        related = extract_related_concepts(keyword, max_concepts=max_concepts)
        
        if related:
            # Group by relation type
            by_relation = {}
            for r in related:
                rel = r["relation"]
                if rel not in by_relation:
                    by_relation[rel] = []
                by_relation[rel].append(r["concept"])
            
            # Format output
            parts = [f"**{keyword}**:"]
            for rel, concepts in by_relation.items():
                parts.append(f"  - {rel}: {', '.join(concepts[:5])}")
            
            context_parts.append("\n".join(parts))
    
    if context_parts:
        return "\n\n".join(context_parts)
    
    return ""


##########################
# Enhanced Tavily Search Tool
##########################

@tool(description="Web search enhanced with ConceptNet knowledge graph for better query understanding and expansion")
async def kg_enhanced_tavily_search(
    queries: List[str],
    use_kg_enhancement: bool = True,
    expansion_strategy: Literal["append", "or", "separate"] = "or",
    max_expansion_terms: int = 3,
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Execute web search with optional ConceptNet knowledge graph enhancement.
    
    This tool wraps Tavily search and optionally enhances queries using
    ConceptNet to find related concepts, synonyms, and domain knowledge.
    
    Args:
        queries: List of search queries to execute
        use_kg_enhancement: Whether to use knowledge graph for query expansion
        expansion_strategy: How to expand queries ("append", "or", "separate")
        max_expansion_terms: Maximum expansion terms per query
        max_results: Maximum results per query
        topic: Search topic category
        config: Runtime configuration
        
    Returns:
        Formatted search results with KG enhancement information
    """
    # Import here to avoid circular imports
    from open_deep_research.utils import tavily_search
    
    enhanced_queries = []
    kg_enhancement_info = []
    
    for query in queries:
        # Decide whether to enhance this query
        should_enhance = use_kg_enhancement and should_enhance_with_kg(query)
        
        if should_enhance:
            # Get KG context for logging
            keywords = [w for w in query.split() if len(w) > 3][:2]
            related_info = []
            
            for kw in keywords:
                related = extract_related_concepts(kw, max_concepts=5)
                if related:
                    concepts = [r["concept"] for r in related[:3]]
                    related_info.append(f"'{kw}' -> [{', '.join(concepts)}]")
            
            if related_info:
                kg_enhancement_info.append(f"Query: {query}")
                kg_enhancement_info.append(f"  KG Expansions: {'; '.join(related_info)}")
    
            # Expand the query
            expanded = expand_query_with_kg(
                query,
                max_expansion_terms=max_expansion_terms,
                expansion_strategy=expansion_strategy
            )
            enhanced_queries.extend(expanded)
        else:
            enhanced_queries.append(query)
    
    # Remove duplicates while preserving order
    seen = set()
    final_queries = []
    for q in enhanced_queries:
        if q not in seen:
            seen.add(q)
            final_queries.append(q)
    
    # Call the original Tavily search
    try:
        search_results = await tavily_search.ainvoke({
            "queries": final_queries,
            "max_results": max_results,
            "topic": topic,
            "config": config
        })
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Search failed: {str(e)}"
    
    # Prepend KG enhancement information with clear formatting for report inclusion
    output_parts = []
    
    if kg_enhancement_info:
        output_parts.append("\n" + "=" * 60)
        output_parts.append("KNOWLEDGE GRAPH ENHANCED SEARCH")
        output_parts.append("=" * 60)
        output_parts.append("\nThis search was enhanced using ConceptNet knowledge graph.")
        output_parts.append("The following query expansions were applied:\n")
        output_parts.extend(kg_enhancement_info)
        output_parts.append(f"\nOriginal queries: {queries}")
        output_parts.append(f"Expanded queries: {final_queries}")
        output_parts.append("\n[Note: Knowledge graph enhancement helps discover related")
        output_parts.append("concepts and improves search coverage for academic research.]")
        output_parts.append("=" * 60 + "\n")
    
    output_parts.append(search_results)
    
    return "\n".join(output_parts)


##########################
# Standalone Query Analysis Tool
##########################

@tool(description="Analyze a query using ConceptNet knowledge graph to find related concepts")
def analyze_query_with_kg(
    query: str,
    max_concepts_per_term: int = 5
) -> str:
    """Analyze a query using ConceptNet to discover related concepts.
    
    Use this tool to understand the semantic context of a query before searching.
    It returns related concepts, synonyms, and knowledge graph relationships.
    
    Args:
        query: The query to analyze
        max_concepts_per_term: Maximum related concepts per query term
        
    Returns:
        Formatted analysis with related concepts and relationships
    """
    keywords = [w for w in query.split() if len(w) > 3]
    
    if not keywords:
        return f"No significant keywords found in query: {query}"
    
    output_parts = [f"=== Knowledge Graph Analysis for: {query} ===\n"]
    
    for keyword in keywords[:3]:  # Analyze top 3 keywords
        output_parts.append(f"\n### Concept: {keyword}")
        
        related = extract_related_concepts(keyword, max_concepts=max_concepts_per_term * 2)
        
        if not related:
            output_parts.append("  No related concepts found in ConceptNet")
            continue
        
        # Group by relation type
        by_relation = {}
        for r in related:
            rel = r["relation"]
            if rel not in by_relation:
                by_relation[rel] = []
            by_relation[rel].append(f"{r['concept']} ({r['weight']:.2f})")
        
        for rel, concepts in by_relation.items():
            output_parts.append(f"  {rel}:")
            for concept in concepts[:max_concepts_per_term]:
                output_parts.append(f"    - {concept}")
    
    # Suggest expanded queries
    output_parts.append("\n### Suggested Expanded Queries:")
    expanded = expand_query_with_kg(query, max_expansion_terms=3, expansion_strategy="separate")
    for i, eq in enumerate(expanded, 1):
        output_parts.append(f"  {i}. {eq}")
    
    return "\n".join(output_parts)


##########################
# Utility Functions for Integration
##########################

def get_kg_enhanced_tools() -> List:
    """Return list of KG-enhanced search tools for registration.
    
    Returns:
        List of tool objects to be added to the agent's toolkit
    """
    return [
        kg_enhanced_tavily_search,
        analyze_query_with_kg
    ]


def clear_conceptnet_cache():
    """Clear the ConceptNet query cache.
    
    Useful for testing or when you want fresh results.
    """
    query_conceptnet_cached.cache_clear()
    logger.info("ConceptNet cache cleared")


def get_cache_stats() -> Dict[str, int]:
    """Get ConceptNet cache statistics.
    
    Returns:
        Dict with 'hits', 'misses', 'size', 'maxsize'
    """
    info = query_conceptnet_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "maxsize": info.maxsize
    }

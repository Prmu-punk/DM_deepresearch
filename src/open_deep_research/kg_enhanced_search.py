"""Knowledge Graph Enhanced Search using ConceptNet and Tavily.

This module provides functionality to enhance Tavily web searches using
ConceptNet knowledge graph for query expansion and concept enrichment.

Features:
- TF-IDF style scoring: prioritizes locally relevant but globally rare concepts
- Phrase-aware matching with tiered fallback
- Relation type weighting for semantic relevance
"""

import asyncio
import logging
import math
import re
import requests
from collections import Counter
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

# ConceptNet API configuration
CONCEPTNET_API_BASE = "http://api.conceptnet.io"
CONCEPTNET_TIMEOUT = 5  # seconds

# TF-IDF configuration
MIN_WORD_LENGTH = 3  # Minimum word length to consider
STOP_NODES = {
    'this', 'that', 'these', 'those', 'study', 'result', 'results', 
    'data', 'method', 'methods', 'analysis', 'research', 'paper',
    'figure', 'table', 'section', 'example', 'case', 'type', 'types'
}

# Relation type weights (semantic relations weighted higher)
RELATION_WEIGHTS = {
    # High value semantic relations
    'is_a': 2.0, 'isa': 2.0, 'isA': 2.0,
    'part_of': 1.8, 'partof': 1.8, 'PartOf': 1.8,
    'has_part': 1.8, 'haspart': 1.8, 'HasPart': 1.8,
    'related_to': 1.5, 'relatedto': 1.5, 'RelatedTo': 1.5,
    'similar_to': 1.5, 'similarto': 1.5, 'SimilarTo': 1.5,
    'synonym': 1.5, 'Synonym': 1.5,
    'used_for': 1.5, 'usedfor': 1.5, 'UsedFor': 1.5,
    'causes': 1.5, 'Causes': 1.5,
    'caused_by': 1.5, 'causedby': 1.5, 'CausedBy': 1.5,
    'contains': 1.3, 'Contains': 1.3,
    'has_property': 1.3, 'hasproperty': 1.3, 'HasProperty': 1.3,
    'defined_as': 1.3, 'definedas': 1.3, 'DefinedAs': 1.3,
    # Lower value relations (geographic, temporal, etc.)
    'is_in': 0.5, 'isin': 0.5, 'IsIn': 0.5,
    'located_in': 0.5, 'locatedin': 0.5, 'LocatedIn': 0.5,
    'at_location': 0.5, 'atlocation': 0.5, 'AtLocation': 0.5,
}

# Logger setup
logger = logging.getLogger(__name__)


##########################
# TF-IDF Style Scoring
##########################

def _compute_tfidf_scores(
    local_counts: Counter,
    global_counts: Counter,
    total_nodes: int,
    relation_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute TF-IDF style scores for neighbor nodes.
    
    Score = LocalFreq × log(TotalNodes / GlobalFreq) × RelationWeight
    
    Args:
        local_counts: Counter of (neighbor, relation) -> count for query node
        global_counts: Counter of neighbor -> total count in graph
        total_nodes: Total number of unique nodes in graph
        relation_weights: Optional dict of relation -> weight multiplier
        
    Returns:
        Dict of neighbor -> score
    """
    scores = {}
    
    for (neighbor, relation), local_freq in local_counts.items():
        # Skip stop nodes
        if neighbor.lower() in STOP_NODES:
            continue
        if len(neighbor) < MIN_WORD_LENGTH:
            continue
            
        global_freq = global_counts.get(neighbor, 1)
        
        # TF-IDF: local frequency × inverse global frequency
        idf = math.log(total_nodes / global_freq) if global_freq > 0 else 0
        base_score = local_freq * idf
        
        # Apply relation weight
        rel_weight = 1.0
        if relation_weights:
            rel_weight = relation_weights.get(relation, 1.0)
        
        final_score = base_score * rel_weight
        
        # Aggregate scores if same neighbor appears with different relations
        if neighbor in scores:
            scores[neighbor] = max(scores[neighbor], final_score)
        else:
            scores[neighbor] = final_score
    
    return scores


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
    max_concepts: int = 10,
    use_tfidf: bool = True
) -> List[Dict[str, Any]]:
    """Extract related concepts from ConceptNet using TF-IDF style scoring.
    
    Args:
        concept: The source concept to find relations for
        relation_types: List of relation types to include (None = all useful types)
        max_concepts: Maximum number of related concepts to return
        use_tfidf: If True, use TF-IDF scoring; if False, use simple neighbor search
        
    Returns:
        List of dicts with 'concept', 'relation', 'weight'/'score' keys
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
    
    # Fetch edges
    data = query_conceptnet_cached(concept, limit=50)
    
    if not data or not data.get("edges"):
        return []
    
    edges = data.get("edges", [])
    
    # Filter edges by relation type
    filtered_edges = [
        e for e in edges 
        if e.get("rel", {}).get("label", "") in relation_types
    ]
    
    if use_tfidf and len(filtered_edges) >= 3:
        return _extract_with_tfidf(concept, filtered_edges, max_concepts)
    else:
        return _extract_simple_neighbors(concept, filtered_edges, max_concepts)


def _extract_with_tfidf(
    concept: str, 
    edges: List[Dict], 
    max_concepts: int
) -> List[Dict[str, Any]]:
    """Extract related concepts using TF-IDF style scoring.
    
    Score = LocalFreq × IDF × RelationWeight
    - LocalFreq: How often this neighbor appears in query's edges
    - IDF: log(total_concepts / global_freq) - rare concepts score higher
    - RelationWeight: Semantic relations weighted higher than geographic
    """
    concept_lower = concept.lower()
    
    # Count local frequencies (neighbors of this concept)
    local_counts = Counter()  # (neighbor, relation) -> count
    all_concepts = set()
    
    for edge in edges:
        start_info = edge.get("start", {})
        end_info = edge.get("end", {})
        
        start_label = start_info.get("label", "").lower()
        end_label = end_info.get("label", "").lower()
        start_lang = start_info.get("language", "en")
        end_lang = end_info.get("language", "en")
        rel_label = edge.get("rel", {}).get("label", "")
        
        # Track all concepts for global frequency estimation
        if start_lang == "en" and start_label:
            all_concepts.add(start_label)
        if end_lang == "en" and end_label:
            all_concepts.add(end_label)
        
        # Count neighbors of the query concept
        if start_lang == "en" and end_lang == "en":
            if start_label == concept_lower and end_label:
                local_counts[(end_label, rel_label)] += 1
            elif end_label == concept_lower and start_label:
                local_counts[(start_label, rel_label)] += 1
    
    if not local_counts:
        return _extract_simple_neighbors(concept, edges, max_concepts)
    
    # For ConceptNet, we use edge count as a proxy for global frequency
    # (concepts that appear in many edges are more common)
    global_counts = Counter()
    for edge in edges:
        start_label = edge.get("start", {}).get("label", "").lower()
        end_label = edge.get("end", {}).get("label", "").lower()
        if start_label:
            global_counts[start_label] += 1
        if end_label:
            global_counts[end_label] += 1
    
    total_nodes = max(len(all_concepts), 1)
    
    # Compute TF-IDF scores with relation weights
    scores = _compute_tfidf_scores(
        local_counts, 
        global_counts, 
        total_nodes, 
        RELATION_WEIGHTS
    )
    
    if not scores:
        return _extract_simple_neighbors(concept, edges, max_concepts)
    
    # Sort by score and return top results
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build result list with relation info
    results = []
    neighbor_to_relation = {n: r for (n, r), _ in local_counts.items()}
    
    for neighbor, score in sorted_results[:max_concepts]:
        if neighbor == concept_lower:
            continue
        results.append({
            "concept": neighbor,
            "relation": neighbor_to_relation.get(neighbor, "RelatedTo"),
            "weight": score
        })
    
    return results


def _extract_simple_neighbors(
    concept: str, 
    edges: List[Dict], 
    max_concepts: int
) -> List[Dict[str, Any]]:
    """Extract related concepts using simple neighbor search (original logic)."""
    
    related = []
    seen_concepts = set()
    concept_lower = concept.lower()
    
    for edge in edges:
        # Extract start and end concepts
        start_info = edge.get("start", {})
        end_info = edge.get("end", {})
        
        start_label = start_info.get("label", "")
        end_label = end_info.get("label", "")
        start_lang = start_info.get("language", "en")
        end_lang = end_info.get("language", "en")
        rel_label = edge.get("rel", {}).get("label", "")
        
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
    expansion_strategy: Literal["append", "or", "separate"] = "append"
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
        query: The search query or research topic (used as-is, no splitting)
        max_concepts: Maximum related concepts to include
        
    Returns:
        Formatted string with KG context information
    """
    # Use query as-is without splitting - preserves phrases
    keyword = query.strip()
    
    if not keyword:
        return ""
    
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
        
        result = "\n".join(parts)
        return result
    
    return ""


def _write_debug_log(function_name: str, query: str, context: str) -> None:
    """Write KG expansion debug info to ./debug.txt"""
    import os
    from datetime import datetime
    
    debug_file = os.path.join(os.path.dirname(__file__), "debug.txt")
    
    try:
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[{datetime.now().isoformat()}] {function_name}\n")
            f.write(f"Query: {query}\n")
            f.write("-" * 40 + "\n")
            f.write("Injected KG Context:\n")
            f.write(context + "\n")
            f.write("=" * 80 + "\n")
    except Exception as e:
        logger.warning(f"Failed to write debug log: {e}")


##########################
# Geoscience Detection & GAKG Expansion
##########################

GEO_KEYWORDS = {
    "climate", "meteorology", "atmosphere", "hydrology", "hydrogeology", "precipitation",
    "geology", "geologic", "geoscience", "geophysics", "tectonic", "earthquake", "seismic",
    "seismology", "fault", "volcano", "volcanic", "magma", "igneous", "sediment", "basin",
    "stratigraphy", "paleoclimate", "glacier", "permafrost", "ocean", "oceanography",
    "coast", "shoreline", "erosion", "landslide", "geothermal", "lithosphere", "crust",
    "mantle", "geochemistry", "geomorphology", "remote sensing", "satellite"
}


def is_geoscience_topic(text: str) -> bool:
    """Heuristic check for geoscience topics using a compact keyword seed set."""
    text_lower = (text or "").lower()
    return any(keyword in text_lower for keyword in GEO_KEYWORDS)


@lru_cache(maxsize=4)
def _load_gakg_frame(parquet_path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load GAKG parquet '{parquet_path}': {exc}")
        return None


# Cache for GAKG global frequency statistics
_gakg_global_freq_cache: Dict[str, Tuple[Counter, int]] = {}


def _get_gakg_global_stats(df: pd.DataFrame, cache_key: str) -> Tuple[Counter, int]:
    """
    Get or compute global frequency statistics for GAKG.
    
    Returns:
        (global_counts Counter, total_unique_nodes)
    """
    if cache_key not in _gakg_global_freq_cache:
        # Count how often each concept appears in the entire graph
        subjects = df['subject'].str.lower().str.strip()
        objects = df['object'].str.lower().str.strip()
        
        global_counts = Counter(subjects) + Counter(objects)
        total_nodes = len(global_counts)
        
        _gakg_global_freq_cache[cache_key] = (global_counts, total_nodes)
        logger.info(f"GAKG stats cached: {total_nodes} unique nodes")
    
    return _gakg_global_freq_cache[cache_key]


def _gakg_expand_with_tfidf(
    keyword: str, 
    df: pd.DataFrame, 
    top_k: int = 10,
    use_relation: bool = True,
    cache_key: str = "default"
) -> List[str]:
    """
    Expand keyword using TF-IDF style scoring on GAKG.
    
    Score = LocalFreq × IDF × RelationWeight
    - LocalFreq: How often this neighbor appears with the query
    - IDF: log(total_nodes / global_freq) - rare concepts score higher
    - RelationWeight: Semantic relations weighted higher
    
    Args:
        keyword: Query keyword (phrase supported)
        df: GAKG DataFrame
        top_k: Number of results to return
        use_relation: If True, include relation in output string
        cache_key: Cache key for global stats
        
    Returns:
        List of expanded query strings
    """
    keyword_lower = keyword.lower().strip()
    
    # Get global statistics
    global_counts, total_nodes = _get_gakg_global_stats(df, cache_key)
    
    # Strategy 1: Exact match
    subj_df = df[df["subject"].str.lower() == keyword_lower]
    obj_df = df[df["object"].str.lower() == keyword_lower]
    
    # Strategy 2: Contains match (if exact match fails)
    if subj_df.empty and obj_df.empty:
        subj_df = df[df["subject"].str.lower().str.contains(keyword_lower, regex=False, na=False)]
        obj_df = df[df["object"].str.lower().str.contains(keyword_lower, regex=False, na=False)]
    
    # Strategy 3: Word-level match (if still empty and keyword has multiple words)
    if subj_df.empty and obj_df.empty and ' ' in keyword_lower:
        words = [w for w in keyword_lower.split() if len(w) > 3]
        if words:
            pattern = '|'.join([re.escape(w) for w in words])
            subj_df = df[df["subject"].str.lower().str.contains(pattern, regex=True, na=False)]
            obj_df = df[df["object"].str.lower().str.contains(pattern, regex=True, na=False)]
    
    if subj_df.empty and obj_df.empty:
        return []
    
    # Count local frequencies with relations
    local_counts = Counter()  # (neighbor, relation) -> count
    neighbor_info = {}  # neighbor -> (relation, direction)
    
    # From subject matches: keyword -> object
    for _, row in subj_df.iterrows():
        obj = str(row['object']).lower().strip()
        rel = str(row.get('relation', 'related'))
        local_counts[(obj, rel)] += 1
        if obj not in neighbor_info:
            neighbor_info[obj] = (rel, 'forward')  # keyword rel object
    
    # From object matches: subject -> keyword
    for _, row in obj_df.iterrows():
        subj = str(row['subject']).lower().strip()
        rel = str(row.get('relation', 'related'))
        local_counts[(subj, rel)] += 1
        if subj not in neighbor_info:
            neighbor_info[subj] = (rel, 'backward')  # subject rel keyword
    
    # Compute TF-IDF scores
    scores = _compute_tfidf_scores(local_counts, global_counts, total_nodes, RELATION_WEIGHTS)
    
    if not scores:
        return []
    
    # Sort by score and build output
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    query_terms = []
    for neighbor, score in sorted_results[:top_k]:
        if neighbor == keyword_lower:
            continue
        
        rel, direction = neighbor_info.get(neighbor, ('related', 'forward'))
        
        if use_relation:
            if direction == 'forward':
                query_terms.append(f"{keyword} {rel} {neighbor}")
            else:
                query_terms.append(f"{neighbor} {rel} {keyword}")
        else:
            query_terms.append(f"{keyword} {neighbor}")
    
    return query_terms


def _gakg_expand_queries(keyword: str, df: pd.DataFrame, top_k: int = 5, use_relation: bool = True) -> List[str]:
    """Simple frequency-based expansion (fallback)."""
    keyword_lower = keyword.lower().strip()
    query_terms: List[str] = []

    eff_df = df[df["subject"].str.lower() == keyword_lower]
    top_effects = (
        eff_df.groupby(["relation", "object"]).size().sort_values(ascending=False).head(top_k).index.tolist()
    )

    cau_df = df[df["object"].str.lower() == keyword_lower]
    top_causes = (
        cau_df.groupby(["subject", "relation"]).size().sort_values(ascending=False).head(top_k).index.tolist()
    )

    for rel, obj in top_effects:
        query_terms.append(f"{keyword} {rel} {obj}" if use_relation else f"{keyword} {obj}")

    for sub, rel in top_causes:
        query_terms.append(f"{sub} {rel} {keyword}" if use_relation else f"{sub} {keyword}")

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for term in query_terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)

    return deduped


def build_gakg_context(
    query: str,
    parquet_path: Optional[str],
    max_concepts: int = 10,
    use_relation: bool = True,
) -> str:
    """Build context strings from GAKG for geoscience topics.

    Uses TF-IDF style scoring to find relevant concepts:
    - Prioritizes concepts that are locally frequent but globally rare
    - Weights semantic relations higher than geographic/temporal relations
    - Filters out common stop nodes

    Returns a formatted block that mirrors the ConceptNet context format so the
    downstream prompt handling remains unchanged.
    """

    if not parquet_path:
        return ""

    df = _load_gakg_frame(parquet_path)
    if df is None or not {"subject", "object", "relation"}.issubset(df.columns):
        return ""

    # Use query as-is without splitting into individual words
    # This preserves phrases like "calcium carbonate", "sedimentary rocks"
    keyword = query.strip()

    # Use TF-IDF based expansion
    expansions = _gakg_expand_with_tfidf(
        keyword, df, 
        top_k=max_concepts, 
        use_relation=use_relation,
        cache_key=parquet_path or "default"
    )

    if not expansions:
        return ""

    lines = ["Geoscience KG expansions (TF-IDF):"]
    for term in expansions:
        lines.append(f"- {term}")
    result = "\n".join(lines)
    return result


##########################
# LLM-based KG Enhancement
##########################

async def generate_kg_keywords_with_llm(query: str, llm: BaseChatModel) -> List[str]:
    """Use LLM to extract high-quality keywords for KG lookup."""
    
    prompt = f"""You are a research assistant. Your task is to extract 1-3 core scientific entities or concepts from the user's query for Knowledge Graph lookup.
    
    Query: "{query}"
    
    Rules:
    1. Extract ONLY specific entities (e.g., "nuclear energy", "climate change", "groundwater").
    2. Ignore generic verbs (investigate, analyze) and nouns (study, impact, potential).
    3. Return a comma-separated list of keywords.
    4. If the query is too abstract, return the most relevant broad topic.
    
    Keywords:"""
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = str(response.content).strip()
        keywords = [k.strip() for k in content.split(",") if k.strip()]
        return keywords[:3]  # Limit to top 3
    except Exception as e:
        logger.warning(f"LLM keyword generation failed: {e}")
        # Fallback to simple extraction
        return [w for w in query.split() if len(w) > 4][:2]

async def filter_kg_results_with_llm(query: str, kg_results: str, llm: BaseChatModel) -> str:
    """Use LLM to filter noise from KG results."""
    
    if not kg_results:
        return ""
        
    prompt = f"""You are a research assistant. I have queried a Knowledge Graph for the topic: "{query}".
    The KG returned some raw results. Some may be noise, but others are valuable scientific context.
    
    Raw KG Results:
    {kg_results}
    
    Task:
    1. Identify concepts that are scientifically relevant to the query or provide useful context.
    2. Filter out obvious noise (e.g., unrelated homonyms like "nuclear is in diamond").
    3. Return a clean, bulleted list of the relevant expansions.
    4. IMPORTANT: If you find ANY potentially relevant concepts, include them. Only return "NO_RELEVANT_CONTEXT" if the results are completely nonsensical or unrelated.
    
    Filtered Results:"""
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = str(response.content).strip()
        if "NO_RELEVANT_CONTEXT" in content:
            return ""
        return content
    except Exception as e:
        logger.warning(f"LLM result filtering failed: {e}")
        return kg_results  # Fallback to raw results


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

"""
High-performance search indexing for the PhD notebook system.
"""

import json
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import re
import math

from ..core.note import Note
from ..utils.logging import get_logger, log_performance


class SearchIndex:
    """
    Fast full-text search index using inverted indexing.
    
    Features:
    - Inverted index for fast text search
    - TF-IDF scoring
    - Phrase search support
    - Field-specific search (title, content, tags)
    - Persistent storage with SQLite
    """
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.logger = get_logger(__name__)
        
        # In-memory index structures
        self.term_index: Dict[str, Dict[str, float]] = defaultdict(dict)  # term -> {doc_id: tf_idf}
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}  # doc_id -> metadata
        self.field_index: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        
        # Statistics
        self.document_count = 0
        self.term_document_frequency: Dict[str, int] = defaultdict(int)
        
        # Load existing index
        self._load_index()
    
    @log_performance("search_index_add")
    def add_document(self, doc_id: str, note: Note) -> None:
        """Add or update a document in the search index."""
        # Remove existing document if present
        if doc_id in self.doc_metadata:
            self.remove_document(doc_id)
        
        # Extract text content
        title_text = note.title
        content_text = note.content
        tags_text = ' '.join(note.frontmatter.tags)
        all_text = f"{title_text} {content_text} {tags_text}"
        
        # Tokenize and process text
        terms = self._tokenize(all_text)
        title_terms = self._tokenize(title_text)
        content_terms = self._tokenize(content_text)
        tag_terms = self._tokenize(tags_text)
        
        # Calculate term frequencies
        term_frequencies = self._calculate_term_frequencies(terms)
        
        # Store document metadata
        self.doc_metadata[doc_id] = {
            'title': note.title,
            'note_type': note.note_type.value,
            'tags': note.frontmatter.tags,
            'created': note.frontmatter.created.isoformat(),
            'updated': note.frontmatter.updated.isoformat(),
            'word_count': len(terms),
        }
        
        # Update term document frequencies
        unique_terms = set(terms)
        for term in unique_terms:
            self.term_document_frequency[term] += 1
        
        # Add to inverted index with field information
        for term, tf in term_frequencies.items():
            # Overall index
            idf = math.log(self.document_count / (self.term_document_frequency[term] + 1))
            tf_idf = tf * idf
            self.term_index[term][doc_id] = tf_idf
            
            # Field-specific indexes
            if term in title_terms:
                title_tf = title_terms.count(term) / len(title_terms) if title_terms else 0
                self.field_index['title'][term][doc_id] = title_tf * idf * 2.0  # Boost title matches
            
            if term in content_terms:
                content_tf = content_terms.count(term) / len(content_terms) if content_terms else 0
                self.field_index['content'][term][doc_id] = content_tf * idf
            
            if term in tag_terms:
                tag_tf = tag_terms.count(term) / len(tag_terms) if tag_terms else 0
                self.field_index['tags'][term][doc_id] = tag_tf * idf * 1.5  # Boost tag matches
        
        self.document_count += 1
        
        self.logger.debug(f"Added document to search index: {doc_id}")
    
    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the search index."""
        if doc_id not in self.doc_metadata:
            return
        
        # Remove from inverted indexes
        terms_to_remove = []
        for term, docs in self.term_index.items():
            if doc_id in docs:
                del docs[doc_id]
                self.term_document_frequency[term] -= 1
                if not docs:  # No more documents contain this term
                    terms_to_remove.append(term)
        
        # Clean up empty terms
        for term in terms_to_remove:
            del self.term_index[term]
            del self.term_document_frequency[term]
        
        # Remove from field indexes
        for field_index in self.field_index.values():
            for term, docs in field_index.items():
                docs.pop(doc_id, None)
        
        # Remove metadata
        del self.doc_metadata[doc_id]
        self.document_count -= 1
        
        self.logger.debug(f"Removed document from search index: {doc_id}")
    
    @log_performance("search_query")
    def search(
        self,
        query: str,
        fields: Optional[List[str]] = None,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search the index for documents matching the query.
        
        Returns:
            List of (doc_id, score, metadata) tuples, ordered by relevance
        """
        if not query.strip():
            return []
        
        # Parse query
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # Calculate document scores
        doc_scores: Dict[str, float] = defaultdict(float)
        
        # Search in specified fields or all fields
        search_fields = fields or ['title', 'content', 'tags']
        
        for term in query_terms:
            # Search in overall index
            if term in self.term_index:
                for doc_id, tf_idf in self.term_index[term].items():
                    doc_scores[doc_id] += tf_idf
            
            # Boost field-specific matches
            for field in search_fields:
                if field in self.field_index and term in self.field_index[field]:
                    for doc_id, score in self.field_index[field][term].items():
                        doc_scores[doc_id] += score
        
        # Apply filters
        if filters:
            doc_scores = self._apply_filters(doc_scores, filters)
        
        # Sort by score and apply limit
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Build result list with metadata
        results = []
        for doc_id, score in sorted_results:
            metadata = self.doc_metadata.get(doc_id, {})
            results.append((doc_id, score, metadata))
        
        self.logger.debug(f"Search query '{query}' returned {len(results)} results")
        return results
    
    def phrase_search(self, phrase: str, limit: int = 50) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for exact phrases."""
        # For now, implement as proximity search
        # Full phrase search would require position indexing
        terms = self._tokenize(phrase)
        return self.search(' '.join(terms), limit=limit)
    
    def suggest_terms(self, partial_term: str, limit: int = 10) -> List[str]:
        """Suggest terms based on partial input."""
        partial_term = partial_term.lower()
        suggestions = []
        
        for term in self.term_index.keys():
            if term.startswith(partial_term):
                suggestions.append(term)
        
        # Sort by document frequency
        suggestions.sort(key=lambda t: self.term_document_frequency[t], reverse=True)
        return suggestions[:limit]
    
    def get_term_statistics(self, term: str) -> Dict[str, Any]:
        """Get statistics for a specific term."""
        term = term.lower()
        
        if term not in self.term_index:
            return {'found': False}
        
        doc_count = len(self.term_index[term])
        total_tf_idf = sum(self.term_index[term].values())
        
        return {
            'found': True,
            'document_count': doc_count,
            'document_frequency': self.term_document_frequency[term],
            'total_tf_idf': total_tf_idf,
            'average_tf_idf': total_tf_idf / doc_count if doc_count > 0 else 0
        }
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get overall index statistics."""
        return {
            'document_count': self.document_count,
            'unique_terms': len(self.term_index),
            'index_size_mb': self._estimate_memory_usage() / (1024 * 1024),
            'average_document_length': (
                sum(meta.get('word_count', 0) for meta in self.doc_metadata.values()) / 
                self.document_count if self.document_count > 0 else 0
            )
        }
    
    def save_index(self) -> None:
        """Save the search index to persistent storage."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use SQLite for persistent storage
            with sqlite3.connect(self.index_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS term_index (
                        term TEXT,
                        doc_id TEXT,
                        tf_idf REAL,
                        PRIMARY KEY (term, doc_id)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS doc_metadata (
                        doc_id TEXT PRIMARY KEY,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS index_stats (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                ''')
                
                # Clear existing data
                conn.execute('DELETE FROM term_index')
                conn.execute('DELETE FROM doc_metadata')
                conn.execute('DELETE FROM index_stats')
                
                # Save term index
                for term, docs in self.term_index.items():
                    for doc_id, tf_idf in docs.items():
                        conn.execute(
                            'INSERT INTO term_index VALUES (?, ?, ?)',
                            (term, doc_id, tf_idf)
                        )
                
                # Save document metadata
                for doc_id, metadata in self.doc_metadata.items():
                    conn.execute(
                        'INSERT INTO doc_metadata VALUES (?, ?)',
                        (doc_id, json.dumps(metadata, default=str))
                    )
                
                # Save statistics
                conn.execute(
                    'INSERT INTO index_stats VALUES (?, ?)',
                    ('document_count', str(self.document_count))
                )
                conn.execute(
                    'INSERT INTO index_stats VALUES (?, ?)',
                    ('term_frequencies', json.dumps(self.term_document_frequency))
                )
                
                conn.commit()
            
            self.logger.info(f"Search index saved to {self.index_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save search index: {e}")
    
    def _load_index(self) -> None:
        """Load the search index from persistent storage."""
        if not self.index_path.exists():
            return
        
        try:
            with sqlite3.connect(self.index_path) as conn:
                # Load term index
                cursor = conn.execute('SELECT term, doc_id, tf_idf FROM term_index')
                for term, doc_id, tf_idf in cursor:
                    self.term_index[term][doc_id] = tf_idf
                
                # Load document metadata
                cursor = conn.execute('SELECT doc_id, metadata FROM doc_metadata')
                for doc_id, metadata_json in cursor:
                    self.doc_metadata[doc_id] = json.loads(metadata_json)
                
                # Load statistics
                cursor = conn.execute('SELECT key, value FROM index_stats')
                stats = dict(cursor.fetchall())
                
                if 'document_count' in stats:
                    self.document_count = int(stats['document_count'])
                
                if 'term_frequencies' in stats:
                    self.term_document_frequency = defaultdict(int, json.loads(stats['term_frequencies']))
            
            self.logger.info(f"Search index loaded from {self.index_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load search index: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable terms."""
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric characters
        text = text.lower()
        terms = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        
        # Filter out very short terms and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_terms = [
            term for term in terms 
            if len(term) > 2 and term not in stop_words
        ]
        
        return filtered_terms
    
    def _calculate_term_frequencies(self, terms: List[str]) -> Dict[str, float]:
        """Calculate term frequencies for a document."""
        if not terms:
            return {}
        
        term_counts = defaultdict(int)
        for term in terms:
            term_counts[term] += 1
        
        doc_length = len(terms)
        return {term: count / doc_length for term, count in term_counts.items()}
    
    def _apply_filters(self, doc_scores: Dict[str, float], filters: Dict[str, Any]) -> Dict[str, float]:
        """Apply filters to search results."""
        filtered_scores = {}
        
        for doc_id, score in doc_scores.items():
            metadata = self.doc_metadata.get(doc_id, {})
            
            # Apply each filter
            include = True
            
            if 'note_type' in filters:
                if metadata.get('note_type') != filters['note_type']:
                    include = False
            
            if 'tags' in filters:
                doc_tags = set(metadata.get('tags', []))
                required_tags = set(filters['tags'])
                if not required_tags.intersection(doc_tags):
                    include = False
            
            if 'date_range' in filters:
                # Implement date filtering
                pass
            
            if include:
                filtered_scores[doc_id] = score
        
        return filtered_scores
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the index."""
        # Rough estimation
        term_index_size = sum(
            len(term.encode()) + len(docs) * (len(max(docs.keys(), key=len).encode()) + 8)
            for term, docs in self.term_index.items()
        )
        
        metadata_size = sum(
            len(doc_id.encode()) + len(json.dumps(metadata).encode())
            for doc_id, metadata in self.doc_metadata.items()
        )
        
        return term_index_size + metadata_size


class SemanticSearchIndex:
    """
    Semantic search using vector embeddings (placeholder for AI integration).
    
    This would integrate with sentence transformers or other embedding models
    in a full implementation with AI dependencies.
    """
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.logger = get_logger(__name__)
        
        # Placeholder - would contain actual embeddings
        self.embeddings: Dict[str, List[float]] = {}
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_document(self, doc_id: str, note: Note) -> None:
        """Add document to semantic index (placeholder)."""
        # In a full implementation, this would:
        # 1. Generate embeddings using a model like sentence-transformers
        # 2. Store embeddings in a vector database like ChromaDB
        # 3. Build efficient similarity search structures
        
        self.doc_metadata[doc_id] = {
            'title': note.title,
            'note_type': note.note_type.value,
            'tags': note.frontmatter.tags,
        }
        
        # Placeholder embedding (would be real embeddings)
        text = f"{note.title} {note.content}"
        self.embeddings[doc_id] = [0.0] * 384  # Typical embedding dimension
        
        self.logger.debug(f"Added document to semantic index: {doc_id}")
    
    def semantic_search(
        self,
        query: str,
        limit: int = 50,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform semantic similarity search (placeholder)."""
        # In a full implementation, this would:
        # 1. Generate query embedding
        # 2. Compute cosine similarity with document embeddings
        # 3. Return most similar documents
        
        # Placeholder implementation
        results = []
        for doc_id, metadata in list(self.doc_metadata.items())[:limit]:
            # Fake similarity score
            similarity = 0.8
            if similarity >= similarity_threshold:
                results.append((doc_id, similarity, metadata))
        
        return results
    
    def find_similar_documents(
        self,
        doc_id: str,
        limit: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find documents similar to a given document (placeholder)."""
        # Would use document embedding to find similar documents
        return []
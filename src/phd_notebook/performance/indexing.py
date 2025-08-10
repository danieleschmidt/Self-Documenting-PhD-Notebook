"""
High-performance search indexing and retrieval system.
"""

import json
import sqlite3
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
import threading
import asyncio

from ..utils.exceptions import MetricsError


class SearchIndex:
    """Full-text search index for notes and content."""
    
    def __init__(self, index_path: Optional[Path] = None):
        if index_path is None:
            index_path = Path.home() / '.phd-notebook' / 'search.db'
        
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self._lock = threading.RLock()
        
        # Text processing
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
            'that', 'the', 'to', 'was', 'will', 'with', 'the'
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for search index."""
        with sqlite3.connect(self.index_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    tags TEXT,
                    note_type TEXT,
                    file_path TEXT,
                    created_at REAL,
                    updated_at REAL,
                    content_hash TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS terms (
                    term TEXT,
                    document_id TEXT,
                    frequency INTEGER,
                    positions TEXT,
                    PRIMARY KEY (term, document_id)
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_terms_doc ON terms(document_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_docs_updated ON documents(updated_at)')
            
            # Enable FTS for better text search
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_documents 
                USING fts5(id, title, content, tags, content='documents', content_rowid='rowid')
            ''')
            
            conn.commit()
    
    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        tags: List[str] = None,
        note_type: str = '',
        file_path: str = '',
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ) -> None:
        """Add or update a document in the search index."""
        tags = tags or []
        created_at = created_at or datetime.now()
        updated_at = updated_at or datetime.now()
        
        # Calculate content hash to detect changes
        content_hash = hashlib.sha256((title + content + str(tags)).encode()).hexdigest()
        
        with self._lock:
            with sqlite3.connect(self.index_path) as conn:
                # Check if document exists and hasn't changed
                existing = conn.execute(
                    'SELECT content_hash FROM documents WHERE id = ?',
                    (doc_id,)
                ).fetchone()
                
                if existing and existing[0] == content_hash:
                    return  # No changes, skip indexing
                
                # Remove old document if it exists
                if existing:
                    self.remove_document(doc_id)
                
                # Insert document
                conn.execute('''
                    INSERT OR REPLACE INTO documents 
                    (id, title, content, tags, note_type, file_path, created_at, updated_at, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id, title, content, json.dumps(tags), 
                    note_type, file_path,
                    created_at.timestamp(), updated_at.timestamp(),
                    content_hash
                ))
                
                # Index terms
                self._index_text_terms(conn, doc_id, title, content, tags)
                
                # Update FTS
                conn.execute('''
                    INSERT INTO fts_documents(id, title, content, tags)
                    VALUES (?, ?, ?, ?)
                ''', (doc_id, title, content, ' '.join(tags)))
                
                conn.commit()
    
    def _index_text_terms(
        self, 
        conn: sqlite3.Connection,
        doc_id: str, 
        title: str, 
        content: str, 
        tags: List[str]
    ) -> None:
        """Extract and index terms from text."""
        # Combine all text
        all_text = f"{title} {content} {' '.join(tags)}"
        
        # Extract terms
        terms = self._extract_terms(all_text)
        term_positions = self._get_term_positions(all_text, terms)
        
        # Count term frequencies
        term_freq = Counter(terms)
        
        # Insert terms
        for term, frequency in term_freq.items():
            positions = term_positions.get(term, [])
            conn.execute('''
                INSERT INTO terms (term, document_id, frequency, positions)
                VALUES (?, ?, ?, ?)
            ''', (term, doc_id, frequency, json.dumps(positions)))
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract searchable terms from text."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Filter stop words and short words
        terms = [
            word for word in words
            if len(word) > 2 and word not in self.stop_words
        ]
        
        return terms
    
    def _get_term_positions(self, text: str, terms: List[str]) -> Dict[str, List[int]]:
        """Get positions of terms in text."""
        text_lower = text.lower()
        positions = defaultdict(list)
        
        for term in set(terms):  # Only process unique terms
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                positions[term].append(pos)
                start = pos + 1
        
        return dict(positions)
    
    def remove_document(self, doc_id: str) -> None:
        """Remove document from search index."""
        with self._lock:
            with sqlite3.connect(self.index_path) as conn:
                conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
                conn.execute('DELETE FROM terms WHERE document_id = ?', (doc_id,))
                conn.execute('DELETE FROM fts_documents WHERE id = ?', (doc_id,))
                conn.commit()
    
    def search(
        self,
        query: str,
        limit: int = 50,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search documents using the index."""
        if not query.strip():
            return []
        
        filters = filters or {}
        
        with self._lock:
            with sqlite3.connect(self.index_path) as conn:
                # Use FTS for basic search
                fts_results = self._fts_search(conn, query, limit * 2, filters)
                
                # If we have term-based search capability, enhance results
                if len(query.split()) > 1:
                    term_results = self._term_search(conn, query, limit * 2, filters)
                    
                    # Merge and rank results
                    combined_results = self._merge_search_results(fts_results, term_results)
                else:
                    combined_results = fts_results
                
                # Apply final filtering and ranking
                return self._rank_results(combined_results, query)[:limit]
    
    def _fts_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search using SQLite FTS."""
        # Prepare FTS query
        fts_query = ' '.join(f'"{term}"' for term in query.split())
        
        # Build base query
        sql = '''
            SELECT d.*, fts.rank
            FROM fts_documents fts
            JOIN documents d ON fts.id = d.id
            WHERE fts_documents MATCH ?
        '''
        params = [fts_query]
        
        # Apply filters
        if filters.get('note_type'):
            sql += ' AND d.note_type = ?'
            params.append(filters['note_type'])
        
        if filters.get('tags'):
            sql += ' AND d.tags LIKE ?'
            params.append(f'%{filters["tags"]}%')
        
        if filters.get('created_after'):
            sql += ' AND d.created_at > ?'
            params.append(filters['created_after'].timestamp())
        
        sql += ' ORDER BY fts.rank LIMIT ?'
        params.append(limit)
        
        rows = conn.execute(sql, params).fetchall()
        
        results = []
        for row in rows:
            result = {
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'tags': json.loads(row[3]),
                'note_type': row[4],
                'file_path': row[5],
                'created_at': datetime.fromtimestamp(row[6]),
                'updated_at': datetime.fromtimestamp(row[7]),
                'score': 1.0  # FTS rank
            }
            results.append(result)
        
        return results
    
    def _term_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search using term frequencies and positions."""
        query_terms = self._extract_terms(query)
        if not query_terms:
            return []
        
        # Find documents containing query terms
        sql = '''
            SELECT t.document_id, SUM(t.frequency) as total_freq, COUNT(*) as term_matches
            FROM terms t
            WHERE t.term IN ({})
            GROUP BY t.document_id
            HAVING term_matches > 0
            ORDER BY term_matches DESC, total_freq DESC
            LIMIT ?
        '''.format(','.join('?' * len(query_terms)))
        
        params = query_terms + [limit]
        doc_scores = conn.execute(sql, params).fetchall()
        
        if not doc_scores:
            return []
        
        # Get document details
        doc_ids = [str(row[0]) for row in doc_scores]
        score_map = {row[0]: (row[1], row[2]) for row in doc_scores}
        
        sql = '''
            SELECT id, title, content, tags, note_type, file_path, created_at, updated_at
            FROM documents
            WHERE id IN ({})
        '''.format(','.join('?' * len(doc_ids)))
        
        rows = conn.execute(sql, doc_ids).fetchall()
        
        results = []
        for row in rows:
            freq, matches = score_map[row[0]]
            score = (matches / len(query_terms)) * 0.7 + (freq / 100) * 0.3
            
            result = {
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'tags': json.loads(row[3]),
                'note_type': row[4],
                'file_path': row[5],
                'created_at': datetime.fromtimestamp(row[6]),
                'updated_at': datetime.fromtimestamp(row[7]),
                'score': score
            }
            results.append(result)
        
        return results
    
    def _merge_search_results(
        self,
        fts_results: List[Dict],
        term_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Merge results from different search methods."""
        seen_ids = set()
        merged = []
        
        # Add FTS results first (they tend to be more relevant)
        for result in fts_results:
            if result['id'] not in seen_ids:
                result['score'] *= 1.2  # Boost FTS results
                merged.append(result)
                seen_ids.add(result['id'])
        
        # Add term results that weren't in FTS results
        for result in term_results:
            if result['id'] not in seen_ids:
                merged.append(result)
                seen_ids.add(result['id'])
        
        return merged
    
    def _rank_results(self, results: List[Dict], query: str) -> List[Dict[str, Any]]:
        """Apply additional ranking to search results."""
        query_lower = query.lower()
        
        for result in results:
            # Boost score based on title matches
            if query_lower in result['title'].lower():
                result['score'] *= 1.5
            
            # Boost recent documents slightly
            days_old = (datetime.now() - result['updated_at']).days
            if days_old < 30:
                result['score'] *= 1.1
            
            # Boost based on content length (longer = more substantial)
            content_length = len(result['content'])
            if content_length > 1000:
                result['score'] *= 1.1
        
        # Sort by score
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def suggest_terms(self, prefix: str, limit: int = 10) -> List[str]:
        """Suggest terms based on prefix for autocomplete."""
        if len(prefix) < 2:
            return []
        
        with self._lock:
            with sqlite3.connect(self.index_path) as conn:
                rows = conn.execute('''
                    SELECT term, SUM(frequency) as total_freq
                    FROM terms
                    WHERE term LIKE ?
                    GROUP BY term
                    ORDER BY total_freq DESC
                    LIMIT ?
                ''', (f'{prefix}%', limit)).fetchall()
                
                return [row[0] for row in rows]
    
    def get_popular_terms(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get most frequently used terms."""
        with self._lock:
            with sqlite3.connect(self.index_path) as conn:
                rows = conn.execute('''
                    SELECT term, SUM(frequency) as total_freq
                    FROM terms
                    GROUP BY term
                    ORDER BY total_freq DESC
                    LIMIT ?
                ''', (limit,)).fetchall()
                
                return [(row[0], row[1]) for row in rows]
    
    def get_related_documents(self, doc_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find documents related to the given document."""
        with self._lock:
            with sqlite3.connect(self.index_path) as conn:
                # Get terms from the source document
                source_terms = conn.execute('''
                    SELECT term FROM terms WHERE document_id = ?
                ''', (doc_id,)).fetchall()
                
                if not source_terms:
                    return []
                
                term_list = [row[0] for row in source_terms]
                
                # Find documents with similar terms
                sql = '''
                    SELECT t.document_id, COUNT(*) as common_terms,
                           AVG(t.frequency) as avg_freq
                    FROM terms t
                    WHERE t.term IN ({}) AND t.document_id != ?
                    GROUP BY t.document_id
                    ORDER BY common_terms DESC, avg_freq DESC
                    LIMIT ?
                '''.format(','.join('?' * len(term_list)))
                
                params = term_list + [doc_id, limit]
                related_docs = conn.execute(sql, params).fetchall()
                
                if not related_docs:
                    return []
                
                # Get document details
                doc_ids = [row[0] for row in related_docs]
                sql = '''
                    SELECT id, title, content, tags, note_type
                    FROM documents
                    WHERE id IN ({})
                '''.format(','.join('?' * len(doc_ids)))
                
                rows = conn.execute(sql, doc_ids).fetchall()
                
                results = []
                for row in rows:
                    result = {
                        'id': row[0],
                        'title': row[1],
                        'content': row[2][:200] + '...' if len(row[2]) > 200 else row[2],
                        'tags': json.loads(row[3]),
                        'note_type': row[4]
                    }
                    results.append(result)
                
                return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        with self._lock:
            with sqlite3.connect(self.index_path) as conn:
                doc_count = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
                term_count = conn.execute('SELECT COUNT(DISTINCT term) FROM terms').fetchone()[0]
                
                # Get most recent update
                recent_update = conn.execute(
                    'SELECT MAX(updated_at) FROM documents'
                ).fetchone()[0]
                
                last_updated = None
                if recent_update:
                    last_updated = datetime.fromtimestamp(recent_update)
                
                return {
                    'total_documents': doc_count,
                    'unique_terms': term_count,
                    'last_updated': last_updated,
                    'index_file_size': self.index_path.stat().st_size
                }
    
    def rebuild_index(self) -> None:
        """Rebuild the entire search index."""
        with self._lock:
            # Clear existing data
            with sqlite3.connect(self.index_path) as conn:
                conn.execute('DELETE FROM terms')
                conn.execute('DELETE FROM fts_documents')
                conn.commit()
            
            # Note: In a full implementation, this would re-index all documents
            # from the vault. For now, this is a placeholder.
            print("Index rebuild completed")


# Global search index instance
_global_search_index = None

def get_search_index(index_path: Optional[Path] = None) -> SearchIndex:
    """Get global search index instance."""
    global _global_search_index
    
    if _global_search_index is None:
        _global_search_index = SearchIndex(index_path)
    
    return _global_search_index
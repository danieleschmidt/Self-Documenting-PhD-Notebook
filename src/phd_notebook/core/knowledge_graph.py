"""
Knowledge Graph for tracking relationships between research concepts and notes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .note import Note, Link


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    title: str
    node_type: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    importance_score: float = 0.0


@dataclass 
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relationship: str
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Knowledge graph for tracking relationships between research concepts.
    
    Features:
    - Automatic link extraction from notes
    - Concept clustering and topic modeling
    - Research gap identification
    - Citation network analysis
    """
    
    def __init__(self, vault_manager=None):
        self.vault = vault_manager
        
        # Graph storage
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # Metrics cache
        self._centrality_cache: Dict[str, float] = {}
        self._cluster_cache: Dict[str, List[str]] = {}
        self._last_analysis: Optional[datetime] = None
        
        # Load existing graph if available
        self._load_graph()
    
    def add_note(self, note: Note) -> None:
        """Add a note to the knowledge graph."""
        node_id = self._generate_node_id(note.title)
        
        # Create or update node
        node = GraphNode(
            id=node_id,
            title=note.title,
            node_type=note.note_type.value,
            tags=note.frontmatter.tags,
            metadata={
                "created": note.frontmatter.created,
                "updated": note.frontmatter.updated,
                "status": getattr(note.frontmatter, 'status', 'unknown'),
                "priority": getattr(note.frontmatter, 'priority', 3),
            }
        )
        
        self.nodes[node_id] = node
        
        # Add edges from links in the note
        for link in note.get_links():
            target_id = self._generate_node_id(link.target)
            self.add_edge(node_id, target_id, link.relationship, link.confidence)
        
        # Invalidate caches
        self._invalidate_caches()
    
    def add_edge(
        self, 
        source: str, 
        target: str, 
        relationship: str = "relates_to",
        weight: float = 1.0,
        **metadata
    ) -> None:
        """Add an edge between two nodes."""
        edge = GraphEdge(
            source=source,
            target=target, 
            relationship=relationship,
            weight=weight,
            metadata=metadata
        )
        
        self.edges.append(edge)
        self.adjacency[source].add(target)
        
        # For undirected relationships, add reverse edge
        if relationship in ["relates_to", "similar_to", "connected_to"]:
            self.adjacency[target].add(source)
    
    def get_connected_notes(self, note_title: str, max_depth: int = 2) -> List[str]:
        """Get notes connected to a given note within max_depth."""
        node_id = self._generate_node_id(note_title)
        
        if node_id not in self.nodes:
            return []
        
        visited = set()
        current_level = {node_id}
        
        for depth in range(max_depth):
            next_level = set()
            
            for node in current_level:
                if node in visited:
                    continue
                    
                visited.add(node)
                next_level.update(self.adjacency[node])
            
            current_level = next_level - visited
            
            if not current_level:
                break
        
        # Convert back to titles and exclude the original note
        connected_titles = []
        for node_id in visited:
            if node_id in self.nodes and self.nodes[node_id].title != note_title:
                connected_titles.append(self.nodes[node_id].title)
        
        return connected_titles
    
    def find_clusters(self, min_cluster_size: int = 3) -> Dict[str, List[str]]:
        """Find clusters of related notes using simple connectivity."""
        if self._cluster_cache and self._is_cache_valid():
            return self._cluster_cache
        
        visited = set()
        clusters = {}
        cluster_id = 0
        
        for node_id in self.nodes:
            if node_id in visited:
                continue
            
            # BFS to find connected component
            cluster = []
            queue = [node_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.append(self.nodes[current].title)
                
                # Add neighbors
                for neighbor in self.adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            # Only keep clusters above minimum size
            if len(cluster) >= min_cluster_size:
                clusters[f"cluster_{cluster_id}"] = cluster
                cluster_id += 1
        
        self._cluster_cache = clusters
        return clusters
    
    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate centrality scores for all nodes."""
        if self._centrality_cache and self._is_cache_valid():
            return self._centrality_cache
        
        centrality = {}
        
        for node_id in self.nodes:
            # Simple degree centrality
            degree = len(self.adjacency[node_id])
            centrality[self.nodes[node_id].title] = degree
        
        # Normalize by max degree
        if centrality:
            max_degree = max(centrality.values())
            if max_degree > 0:
                centrality = {k: v / max_degree for k, v in centrality.items()}
        
        self._centrality_cache = centrality
        return centrality
    
    def identify_research_gaps(self) -> List[Dict[str, Any]]:
        """Identify potential research gaps in the knowledge graph."""
        clusters = self.find_clusters()
        centrality = self.calculate_centrality()
        
        gaps = []
        
        # Find bridge opportunities between clusters
        cluster_list = list(clusters.values())
        for i, cluster1 in enumerate(cluster_list):
            for j, cluster2 in enumerate(cluster_list[i+1:], i+1):
                # Check if clusters are connected
                connections = self._count_inter_cluster_connections(cluster1, cluster2)
                
                if connections < 2:  # Weak connection = potential gap
                    gap = {
                        "type": "bridge_opportunity",
                        "description": f"Bridge between {clusters[f'cluster_{i}']}and {clusters[f'cluster_{j}']}",
                        "cluster1": cluster1[:3],  # Show first 3 notes
                        "cluster2": cluster2[:3],
                        "potential_impact": "high" if connections == 0 else "medium"
                    }
                    gaps.append(gap)
        
        # Find isolated high-importance nodes
        for node_id, node in self.nodes.items():
            if (len(self.adjacency[node_id]) <= 1 and 
                node.metadata.get("priority", 3) >= 4):
                
                gap = {
                    "type": "isolated_important_concept",
                    "description": f"Important concept '{node.title}' lacks connections",
                    "node": node.title,
                    "suggestions": "Consider relating to similar work or expanding research"
                }
                gaps.append(gap)
        
        return gaps
    
    def get_research_trajectory(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Analyze research trajectory over time."""
        # Sort nodes by creation date
        timeline = []
        
        for node in self.nodes.values():
            created = node.metadata.get("created", node.created_at)
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace('Z', '+00:00'))
            
            if start_date and created < start_date:
                continue
            if end_date and created > end_date:
                continue
            
            timeline.append({
                "date": created,
                "title": node.title,
                "type": node.node_type,
                "tags": node.tags
            })
        
        # Sort by date
        timeline.sort(key=lambda x: x["date"])
        
        return timeline
    
    def suggest_connections(self, note_title: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Suggest potential connections for a note."""
        node_id = self._generate_node_id(note_title)
        
        if node_id not in self.nodes:
            return []
        
        current_node = self.nodes[node_id]
        suggestions = []
        
        # Find nodes with similar tags
        current_tags = set(current_node.tags)
        
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
            
            # Skip if already connected
            if other_id in self.adjacency[node_id]:
                continue
            
            other_tags = set(other_node.tags)
            tag_overlap = len(current_tags.intersection(other_tags))
            
            if tag_overlap > 0:
                score = tag_overlap / len(current_tags.union(other_tags))
                
                suggestions.append({
                    "target": other_node.title,
                    "reason": f"Shares {tag_overlap} tags: {list(current_tags.intersection(other_tags))}",
                    "confidence": score,
                    "suggested_relationship": "relates_to"
                })
        
        # Sort by confidence and limit results
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:limit]
    
    def export_graph(self, format: str = "json") -> str:
        """Export the knowledge graph in various formats."""
        if format == "json":
            return self._export_json()
        elif format == "dot":
            return self._export_graphviz()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export graph as JSON."""
        export_data = {
            "nodes": [
                {
                    "id": node.id,
                    "title": node.title,
                    "type": node.node_type,
                    "tags": node.tags,
                    "metadata": node.metadata
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relationship,
                    "weight": edge.weight
                }
                for edge in self.edges
            ],
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "created_at": datetime.now().isoformat()
            }
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_graphviz(self) -> str:
        """Export graph as Graphviz DOT format."""
        lines = ["digraph KnowledgeGraph {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=rounded];")
        
        # Add nodes
        for node in self.nodes.values():
            label = node.title.replace('"', '\\"')
            color = self._get_node_color(node.node_type)
            lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}", style="filled,rounded"];')
        
        # Add edges
        for edge in self.edges:
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.relationship}"];')
        
        lines.append("}")
        return "\n".join(lines)
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        color_map = {
            "experiment": "lightblue",
            "literature": "lightgreen", 
            "idea": "lightyellow",
            "meeting": "lightpink",
            "project": "lightcoral",
            "analysis": "lightcyan",
        }
        return color_map.get(node_type, "lightgray")
    
    def _generate_node_id(self, title: str) -> str:
        """Generate a consistent node ID from title."""
        return title.lower().replace(" ", "_").replace("-", "_")
    
    def _count_inter_cluster_connections(self, cluster1: List[str], cluster2: List[str]) -> int:
        """Count connections between two clusters."""
        connections = 0
        
        for note1 in cluster1:
            node1_id = self._generate_node_id(note1)
            for note2 in cluster2:
                node2_id = self._generate_node_id(note2)
                if node2_id in self.adjacency[node1_id]:
                    connections += 1
        
        return connections
    
    def _invalidate_caches(self) -> None:
        """Invalidate analysis caches."""
        self._centrality_cache.clear()
        self._cluster_cache.clear()
        self._last_analysis = None
    
    def _is_cache_valid(self, max_age_minutes: int = 30) -> bool:
        """Check if cache is still valid."""
        if not self._last_analysis:
            return False
        
        age = datetime.now() - self._last_analysis
        return age.total_seconds() < (max_age_minutes * 60)
    
    def _load_graph(self) -> None:
        """Load existing graph from storage."""
        if not self.vault:
            return
        
        graph_file = self.vault.vault_path / ".obsidian" / "knowledge_graph.json"
        
        if graph_file.exists():
            try:
                data = json.loads(graph_file.read_text())
                
                # Load nodes
                for node_data in data.get("nodes", []):
                    node = GraphNode(**node_data)
                    self.nodes[node.id] = node
                
                # Load edges
                for edge_data in data.get("edges", []):
                    edge = GraphEdge(**edge_data)
                    self.edges.append(edge)
                    self.adjacency[edge.source].add(edge.target)
                
                print(f"📊 Loaded knowledge graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
                
            except Exception as e:
                print(f"⚠️ Could not load knowledge graph: {e}")
    
    def save_graph(self) -> None:
        """Save graph to storage."""
        if not self.vault:
            return
        
        graph_file = self.vault.vault_path / ".obsidian" / "knowledge_graph.json"
        graph_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            graph_file.write_text(self._export_json())
            print("💾 Knowledge graph saved")
        except Exception as e:
            print(f"⚠️ Could not save knowledge graph: {e}")
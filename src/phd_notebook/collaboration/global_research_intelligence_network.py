"""
Global Research Intelligence Network - Generation 3 Enhancement
Worldwide collaborative research intelligence with distributed AI and knowledge sharing.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
from collections import defaultdict, deque
import statistics

# Import numpy with fallback
try:
    import numpy as np
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.fallbacks import np
from utils.fallbacks import nx_module as nx
import hashlib
import networkx as nx

logger = logging.getLogger(__name__)


class NetworkNodeType(Enum):
    """Types of nodes in the research network."""
    RESEARCH_INSTITUTION = "research_institution"
    INDIVIDUAL_RESEARCHER = "individual_researcher"
    INDUSTRY_PARTNER = "industry_partner"
    FUNDING_AGENCY = "funding_agency"
    PUBLICATION_VENUE = "publication_venue"
    DATA_REPOSITORY = "data_repository"
    COLLABORATIVE_PROJECT = "collaborative_project"
    AI_ASSISTANT = "ai_assistant"


class CollaborationStrength(Enum):
    """Strength levels of collaborations."""
    WEAK = "weak"           # Occasional interaction
    MODERATE = "moderate"   # Regular interaction
    STRONG = "strong"       # Frequent collaboration
    INTENSIVE = "intensive" # Deep integration


class KnowledgeType(Enum):
    """Types of knowledge in the network."""
    RESEARCH_FINDING = "research_finding"
    METHODOLOGY = "methodology"
    DATA_INSIGHT = "data_insight"
    THEORETICAL_FRAMEWORK = "theoretical_framework"
    EXPERIMENTAL_PROTOCOL = "experimental_protocol"
    COLLABORATION_PATTERN = "collaboration_pattern"
    FUNDING_OPPORTUNITY = "funding_opportunity"
    RESOURCE_AVAILABILITY = "resource_availability"


@dataclass
class ResearchNode:
    """Node in the global research network."""
    node_id: str
    name: str
    node_type: NetworkNodeType
    location: Tuple[float, float]  # latitude, longitude
    research_areas: List[str]
    expertise_level: Dict[str, float]  # area -> expertise score
    collaboration_capacity: int
    active_projects: List[str]
    network_reputation: float
    last_activity: datetime
    contact_info: Dict[str, str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class KnowledgeElement:
    """Element of knowledge in the network."""
    knowledge_id: str
    title: str
    knowledge_type: KnowledgeType
    content: str
    authors: List[str]
    research_areas: List[str]
    creation_date: datetime
    last_updated: datetime
    citation_count: int
    access_count: int
    quality_score: float
    verification_status: str
    linked_elements: List[str] = None
    
    def __post_init__(self):
        if self.linked_elements is None:
            self.linked_elements = []


@dataclass
class CollaborationLink:
    """Link between nodes representing collaboration."""
    link_id: str
    source_node: str
    target_node: str
    collaboration_strength: CollaborationStrength
    collaboration_type: str
    start_date: datetime
    end_date: Optional[datetime]
    joint_projects: List[str]
    shared_resources: List[str]
    collaboration_outcomes: List[str]
    trust_score: float
    communication_frequency: int  # interactions per month
    
    
@dataclass
class ResearchOpportunity:
    """Research collaboration opportunity."""
    opportunity_id: str
    title: str
    description: str
    research_areas: List[str]
    required_expertise: Dict[str, float]
    funding_available: float
    duration: timedelta
    priority: float
    potential_participants: List[str]
    application_deadline: datetime
    opportunity_type: str  # "funding", "collaboration", "data_sharing", etc.


@dataclass
class IntelligenceInsight:
    """Network intelligence insight."""
    insight_id: str
    title: str
    description: str
    insight_type: str
    confidence: float
    affected_nodes: List[str]
    supporting_evidence: List[str]
    recommendations: List[str]
    generated_at: datetime
    expiration_date: Optional[datetime]
    impact_score: float


class GlobalResearchIntelligenceNetwork:
    """
    Global network for collaborative research intelligence.
    
    Features:
    - Worldwide research network mapping
    - Intelligent collaboration matching
    - Knowledge sharing and synthesis
    - Opportunity discovery and recommendation
    - Network analytics and insights
    - Cross-cultural research facilitation
    - Distributed AI coordination
    - Real-time research intelligence
    """
    
    def __init__(self, network_id: str = None):
        self.network_id = network_id or f"grin_{uuid.uuid4().hex[:8]}"
        
        # Network components
        self.research_nodes: Dict[str, ResearchNode] = {}
        self.knowledge_elements: Dict[str, KnowledgeElement] = {}
        self.collaboration_links: Dict[str, CollaborationLink] = {}
        self.research_opportunities: Dict[str, ResearchOpportunity] = {}
        self.intelligence_insights: Dict[str, IntelligenceInsight] = {}
        
        # Network graph
        self.collaboration_graph = nx.Graph()
        self.knowledge_graph = nx.DiGraph()
        
        # Intelligence engines
        self.collaboration_matcher = CollaborationMatcher()
        self.opportunity_discoverer = OpportunityDiscoverer()
        self.knowledge_synthesizer = KnowledgeSynthesizer()
        self.network_analyzer = NetworkAnalyzer()
        self.reputation_engine = ReputationEngine()
        
        # Distributed AI coordination
        self.ai_coordinator = DistributedAICoordinator()
        self.consensus_engine = ConsensusEngine()
        
        # Network metrics
        self.metrics = {
            "total_nodes": 0,
            "active_collaborations": 0,
            "knowledge_elements": 0,
            "successful_matches": 0,
            "network_density": 0.0,
            "knowledge_flow_rate": 0.0,
            "global_research_impact": 0.0,
            "cross_cultural_collaborations": 0,
            "ai_coordination_efficiency": 0.0
        }
        
        # Network policies and governance
        self.governance_rules = self._initialize_governance_rules()
        
        logger.info(f"Initialized Global Research Intelligence Network: {self.network_id}")
    
    async def add_research_node(self, node: ResearchNode) -> str:
        """Add a new research node to the network."""
        try:
            # Validate node data
            await self._validate_node_data(node)
            
            # Add to network
            self.research_nodes[node.node_id] = node
            self.collaboration_graph.add_node(node.node_id, node_data=node)
            
            # Update network metrics
            self.metrics["total_nodes"] = len(self.research_nodes)
            
            # Discover potential collaborations
            potential_collaborations = await self.collaboration_matcher.find_potential_collaborations(
                node, self.research_nodes
            )
            
            # Store collaboration suggestions
            for suggestion in potential_collaborations:
                await self._store_collaboration_suggestion(suggestion)
            
            # Update reputation
            await self.reputation_engine.initialize_node_reputation(node)
            
            logger.info(f"Added research node: {node.name} ({node.node_type.value})")
            return node.node_id
            
        except Exception as e:
            logger.error(f"Failed to add research node: {e}")
            raise
    
    async def create_collaboration_link(self, 
                                      source_node_id: str,
                                      target_node_id: str,
                                      collaboration_data: Dict[str, Any]) -> str:
        """Create a collaboration link between two nodes."""
        try:
            if source_node_id not in self.research_nodes or target_node_id not in self.research_nodes:
                raise ValueError("One or both nodes not found in network")
            
            link = CollaborationLink(
                link_id=f"collab_{uuid.uuid4().hex[:8]}",
                source_node=source_node_id,
                target_node=target_node_id,
                collaboration_strength=CollaborationStrength(collaboration_data.get("strength", "moderate")),
                collaboration_type=collaboration_data.get("type", "research"),
                start_date=datetime.now(),
                end_date=collaboration_data.get("end_date"),
                joint_projects=collaboration_data.get("joint_projects", []),
                shared_resources=collaboration_data.get("shared_resources", []),
                collaboration_outcomes=collaboration_data.get("outcomes", []),
                trust_score=collaboration_data.get("trust_score", 0.7),
                communication_frequency=collaboration_data.get("communication_frequency", 5)
            )
            
            # Store collaboration link
            self.collaboration_links[link.link_id] = link
            
            # Add edge to collaboration graph
            self.collaboration_graph.add_edge(
                source_node_id, target_node_id,
                link_data=link
            )
            
            # Update metrics
            self.metrics["active_collaborations"] = len(self.collaboration_links)
            self.metrics["network_density"] = nx.density(self.collaboration_graph)
            
            # Update reputation based on new collaboration
            await self.reputation_engine.update_collaboration_reputation(link)
            
            # Check if this is a cross-cultural collaboration
            source_node = self.research_nodes[source_node_id]
            target_node = self.research_nodes[target_node_id]
            
            if self._is_cross_cultural_collaboration(source_node, target_node):
                self.metrics["cross_cultural_collaborations"] += 1
            
            logger.info(f"Created collaboration link: {source_node_id} <-> {target_node_id}")
            return link.link_id
            
        except Exception as e:
            logger.error(f"Failed to create collaboration link: {e}")
            raise
    
    async def share_knowledge(self, knowledge: KnowledgeElement) -> str:
        """Share knowledge element in the network."""
        try:
            # Validate knowledge element
            await self._validate_knowledge_element(knowledge)
            
            # Add to knowledge store
            self.knowledge_elements[knowledge.knowledge_id] = knowledge
            
            # Add to knowledge graph
            self.knowledge_graph.add_node(knowledge.knowledge_id, knowledge_data=knowledge)
            
            # Link to related knowledge elements
            related_elements = await self.knowledge_synthesizer.find_related_knowledge(
                knowledge, self.knowledge_elements
            )
            
            for related_id, relationship_strength in related_elements.items():
                self.knowledge_graph.add_edge(
                    knowledge.knowledge_id, related_id,
                    relationship_strength=relationship_strength
                )
            
            # Notify relevant network nodes
            interested_nodes = await self._find_interested_nodes(knowledge)
            for node_id in interested_nodes:
                await self._notify_node_about_knowledge(node_id, knowledge)
            
            # Update metrics
            self.metrics["knowledge_elements"] = len(self.knowledge_elements)
            self.metrics["knowledge_flow_rate"] = await self._calculate_knowledge_flow_rate()
            
            logger.info(f"Shared knowledge: {knowledge.title}")
            return knowledge.knowledge_id
            
        except Exception as e:
            logger.error(f"Failed to share knowledge: {e}")
            raise
    
    async def discover_research_opportunities(self, 
                                            research_focus: List[str],
                                            node_id: str = None) -> List[ResearchOpportunity]:
        """Discover research opportunities relevant to focus areas."""
        try:
            opportunities = await self.opportunity_discoverer.discover_opportunities(
                research_focus, self.research_nodes, self.knowledge_elements
            )
            
            # Filter by relevance and feasibility
            relevant_opportunities = []
            for opportunity in opportunities:
                relevance_score = await self._calculate_opportunity_relevance(
                    opportunity, research_focus, node_id
                )
                
                if relevance_score > 0.6:  # Threshold for relevance
                    opportunity.priority = relevance_score
                    relevant_opportunities.append(opportunity)
                    
                    # Store opportunity
                    self.research_opportunities[opportunity.opportunity_id] = opportunity
            
            # Sort by priority
            relevant_opportunities.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Discovered {len(relevant_opportunities)} research opportunities")
            return relevant_opportunities
            
        except Exception as e:
            logger.error(f"Failed to discover research opportunities: {e}")
            return []
    
    async def generate_network_intelligence(self) -> List[IntelligenceInsight]:
        """Generate intelligence insights about the research network."""
        try:
            insights = []
            
            # Collaboration pattern analysis
            collaboration_insights = await self.network_analyzer.analyze_collaboration_patterns(
                self.collaboration_graph, self.research_nodes
            )
            insights.extend(collaboration_insights)
            
            # Knowledge flow analysis
            knowledge_flow_insights = await self.network_analyzer.analyze_knowledge_flows(
                self.knowledge_graph, self.knowledge_elements
            )
            insights.extend(knowledge_flow_insights)
            
            # Emerging research trends
            trend_insights = await self.knowledge_synthesizer.identify_emerging_trends(
                self.knowledge_elements
            )
            insights.extend(trend_insights)
            
            # Network bottlenecks and opportunities
            bottleneck_insights = await self.network_analyzer.identify_network_bottlenecks(
                self.collaboration_graph
            )
            insights.extend(bottleneck_insights)
            
            # Cross-cultural collaboration opportunities
            cultural_insights = await self._analyze_cross_cultural_opportunities()
            insights.extend(cultural_insights)
            
            # Store insights
            for insight in insights:
                self.intelligence_insights[insight.insight_id] = insight
            
            logger.info(f"Generated {len(insights)} network intelligence insights")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate network intelligence: {e}")
            return []
    
    async def coordinate_distributed_ai(self, 
                                      ai_task: Dict[str, Any],
                                      participating_nodes: List[str]) -> Dict[str, Any]:
        """Coordinate distributed AI computation across network nodes."""
        try:
            coordination_result = await self.ai_coordinator.coordinate_distributed_task(
                ai_task, participating_nodes, self.research_nodes
            )
            
            # Achieve consensus on results
            consensus_result = await self.consensus_engine.achieve_consensus(
                coordination_result["node_results"], participating_nodes
            )
            
            # Update coordination efficiency metric
            efficiency = coordination_result.get("efficiency", 0.0)
            self.metrics["ai_coordination_efficiency"] = (
                self.metrics["ai_coordination_efficiency"] * 0.9 + efficiency * 0.1
            )
            
            return {
                "task_id": ai_task.get("task_id", "unknown"),
                "coordination_result": coordination_result,
                "consensus_result": consensus_result,
                "participating_nodes": participating_nodes,
                "efficiency": efficiency,
                "completed_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to coordinate distributed AI: {e}")
            return {}
    
    async def facilitate_global_collaboration(self, 
                                            collaboration_request: Dict[str, Any]) -> Dict[str, Any]:
        """Facilitate global research collaboration."""
        try:
            # Extract collaboration requirements
            research_areas = collaboration_request.get("research_areas", [])
            required_expertise = collaboration_request.get("required_expertise", {})
            geographic_preferences = collaboration_request.get("geographic_preferences", [])
            language_requirements = collaboration_request.get("language_requirements", [])
            
            # Find potential collaborators
            potential_collaborators = await self.collaboration_matcher.find_global_collaborators(
                research_areas, required_expertise, geographic_preferences, 
                language_requirements, self.research_nodes
            )
            
            # Assess collaboration feasibility
            feasibility_assessments = []
            for collaborator in potential_collaborators:
                assessment = await self._assess_collaboration_feasibility(
                    collaboration_request, collaborator
                )
                feasibility_assessments.append(assessment)
            
            # Create collaboration proposals
            proposals = await self._create_collaboration_proposals(
                collaboration_request, feasibility_assessments
            )
            
            # Cultural compatibility analysis
            cultural_analysis = await self._analyze_cultural_compatibility(potential_collaborators)
            
            return {
                "potential_collaborators": potential_collaborators,
                "feasibility_assessments": feasibility_assessments,
                "collaboration_proposals": proposals,
                "cultural_analysis": cultural_analysis,
                "recommendation_score": await self._calculate_recommendation_score(proposals),
                "next_steps": await self._generate_collaboration_next_steps(proposals)
            }
            
        except Exception as e:
            logger.error(f"Failed to facilitate global collaboration: {e}")
            return {}
    
    async def synthesize_global_knowledge(self, 
                                        research_question: str) -> Dict[str, Any]:
        """Synthesize knowledge from across the global network."""
        try:
            synthesis_result = await self.knowledge_synthesizer.synthesize_global_knowledge(
                research_question, self.knowledge_elements, self.research_nodes
            )
            
            return {
                "research_question": research_question,
                "synthesized_knowledge": synthesis_result["synthesis"],
                "contributing_sources": synthesis_result["sources"],
                "confidence_score": synthesis_result["confidence"],
                "knowledge_gaps": synthesis_result["gaps"],
                "research_recommendations": synthesis_result["recommendations"],
                "cross_cultural_perspectives": synthesis_result["cultural_perspectives"],
                "synthesis_timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to synthesize global knowledge: {e}")
            return {}
    
    async def _validate_node_data(self, node: ResearchNode) -> None:
        """Validate research node data."""
        required_fields = ["node_id", "name", "node_type", "research_areas"]
        for field in required_fields:
            if not hasattr(node, field) or not getattr(node, field):
                raise ValueError(f"Required field '{field}' missing or empty")
        
        # Validate geographic location
        lat, lon = node.location
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Invalid geographic coordinates")
        
        # Validate expertise levels
        for area, level in node.expertise_level.items():
            if not (0.0 <= level <= 1.0):
                raise ValueError(f"Invalid expertise level for {area}: {level}")
    
    async def _validate_knowledge_element(self, knowledge: KnowledgeElement) -> None:
        """Validate knowledge element data."""
        required_fields = ["knowledge_id", "title", "knowledge_type", "content", "authors"]
        for field in required_fields:
            if not hasattr(knowledge, field) or not getattr(knowledge, field):
                raise ValueError(f"Required field '{field}' missing or empty")
        
        # Validate quality score
        if not (0.0 <= knowledge.quality_score <= 1.0):
            raise ValueError(f"Invalid quality score: {knowledge.quality_score}")
    
    def _is_cross_cultural_collaboration(self, node_a: ResearchNode, node_b: ResearchNode) -> bool:
        """Check if collaboration is cross-cultural based on geographic distance."""
        lat1, lon1 = node_a.location
        lat2, lon2 = node_b.location
        
        # Simple distance calculation (haversine would be more accurate)
        distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5
        
        # Consider cross-cultural if distance > 30 degrees (~3000km)
        return distance > 30
    
    async def _find_interested_nodes(self, knowledge: KnowledgeElement) -> List[str]:
        """Find nodes that might be interested in a knowledge element."""
        interested_nodes = []
        
        for node_id, node in self.research_nodes.items():
            # Check research area overlap
            area_overlap = set(knowledge.research_areas) & set(node.research_areas)
            
            if area_overlap:
                # Calculate interest score based on expertise
                interest_score = 0.0
                for area in area_overlap:
                    expertise = node.expertise_level.get(area, 0.0)
                    interest_score += expertise
                
                if interest_score > 0.5:  # Threshold for interest
                    interested_nodes.append(node_id)
        
        return interested_nodes
    
    def _initialize_governance_rules(self) -> Dict[str, Any]:
        """Initialize network governance rules."""
        return {
            "knowledge_sharing": {
                "open_access_encouraged": True,
                "attribution_required": True,
                "quality_threshold": 0.6,
                "peer_review_required": False
            },
            "collaboration": {
                "minimum_trust_score": 0.5,
                "maximum_project_duration": timedelta(days=1095),  # 3 years
                "resource_sharing_protocols": "fair_use",
                "dispute_resolution": "consensus_based"
            },
            "reputation": {
                "reputation_decay_rate": 0.95,  # Monthly decay
                "minimum_reputation": 0.0,
                "reputation_boost_factors": ["successful_collaboration", "knowledge_contribution"],
                "reputation_penalty_factors": ["failed_collaboration", "low_quality_contribution"]
            },
            "privacy": {
                "personal_data_protection": True,
                "research_data_anonymization": True,
                "cross_border_data_transfers": "gdpr_compliant"
            }
        }
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get comprehensive network metrics."""
        return {
            "network_metrics": self.metrics,
            "network_size": {
                "total_nodes": len(self.research_nodes),
                "node_types": self._get_node_type_distribution(),
                "geographic_distribution": self._get_geographic_distribution(),
                "research_areas": self._get_research_area_distribution()
            },
            "collaboration_metrics": {
                "active_collaborations": len(self.collaboration_links),
                "collaboration_density": nx.density(self.collaboration_graph),
                "average_collaboration_strength": self._get_average_collaboration_strength(),
                "cross_cultural_ratio": self.metrics["cross_cultural_collaborations"] / max(len(self.collaboration_links), 1)
            },
            "knowledge_metrics": {
                "total_knowledge_elements": len(self.knowledge_elements),
                "knowledge_types": self._get_knowledge_type_distribution(),
                "average_quality_score": self._get_average_knowledge_quality(),
                "citation_network_density": nx.density(self.knowledge_graph) if self.knowledge_graph.number_of_nodes() > 0 else 0.0
            },
            "network_health": self._assess_network_health()
        }
    
    def _get_node_type_distribution(self) -> Dict[str, int]:
        """Get distribution of node types."""
        distribution = defaultdict(int)
        for node in self.research_nodes.values():
            distribution[node.node_type.value] += 1
        return dict(distribution)
    
    def _get_geographic_distribution(self) -> Dict[str, int]:
        """Get geographic distribution of nodes."""
        # Simplified continent assignment based on latitude/longitude
        distribution = defaultdict(int)
        
        for node in self.research_nodes.values():
            lat, lon = node.location
            continent = self._get_continent(lat, lon)
            distribution[continent] += 1
        
        return dict(distribution)
    
    def _get_continent(self, lat: float, lon: float) -> str:
        """Get continent based on coordinates (simplified)."""
        if lat > 60:
            return "Arctic"
        elif lat > 30:
            if lon < -60:
                return "North America"
            elif lon < 60:
                return "Europe"
            else:
                return "Asia"
        elif lat > -30:
            if lon < -90:
                return "Americas"
            elif lon < 50:
                return "Africa"
            else:
                return "Asia"
        else:
            if lon < -60:
                return "South America"
            elif lon < 140:
                return "Africa/Antarctica"
            else:
                return "Oceania"
    
    def _assess_network_health(self) -> str:
        """Assess overall network health."""
        health_factors = [
            self.metrics["network_density"] > 0.1,  # Good connectivity
            len(self.research_nodes) > 10,          # Sufficient size
            self.metrics["knowledge_flow_rate"] > 0.5,  # Active knowledge sharing
            self.metrics["ai_coordination_efficiency"] > 0.7,  # Efficient coordination
            self.metrics["cross_cultural_collaborations"] > 0  # Cultural diversity
        ]
        
        health_score = sum(health_factors) / len(health_factors)
        
        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.6:
            return "good"
        elif health_score >= 0.4:
            return "fair"
        else:
            return "needs_improvement"


# Supporting classes for network intelligence

class CollaborationMatcher:
    """Advanced collaboration matching algorithms."""
    
    async def find_potential_collaborations(self, 
                                          new_node: ResearchNode,
                                          existing_nodes: Dict[str, ResearchNode]) -> List[Dict[str, Any]]:
        """Find potential collaborations for a new node."""
        suggestions = []
        
        for node_id, existing_node in existing_nodes.items():
            if node_id == new_node.node_id:
                continue
            
            # Calculate collaboration score
            collaboration_score = await self._calculate_collaboration_score(new_node, existing_node)
            
            if collaboration_score > 0.6:  # Threshold for potential collaboration
                suggestion = {
                    "target_node_id": node_id,
                    "target_node_name": existing_node.name,
                    "collaboration_score": collaboration_score,
                    "shared_areas": list(set(new_node.research_areas) & set(existing_node.research_areas)),
                    "complementary_expertise": await self._find_complementary_expertise(new_node, existing_node),
                    "geographic_distance": self._calculate_distance(new_node.location, existing_node.location),
                    "suggested_collaboration_type": await self._suggest_collaboration_type(new_node, existing_node)
                }
                suggestions.append(suggestion)
        
        # Sort by collaboration score
        suggestions.sort(key=lambda x: x["collaboration_score"], reverse=True)
        
        return suggestions[:10]  # Return top 10 suggestions
    
    async def find_global_collaborators(self,
                                      research_areas: List[str],
                                      required_expertise: Dict[str, float],
                                      geographic_preferences: List[str],
                                      language_requirements: List[str],
                                      nodes: Dict[str, ResearchNode]) -> List[Dict[str, Any]]:
        """Find global collaborators based on specific requirements."""
        matching_collaborators = []
        
        for node_id, node in nodes.items():
            # Check research area compatibility
            area_match = len(set(research_areas) & set(node.research_areas)) / len(research_areas)
            
            if area_match < 0.3:  # Minimum area match threshold
                continue
            
            # Check expertise requirements
            expertise_match = 0.0
            for area, required_level in required_expertise.items():
                node_expertise = node.expertise_level.get(area, 0.0)
                if node_expertise >= required_level:
                    expertise_match += 1.0
            expertise_match /= max(len(required_expertise), 1)
            
            # Geographic preference matching (simplified)
            geographic_match = 1.0  # Default to full match
            if geographic_preferences:
                node_region = self._get_node_region(node.location)
                geographic_match = 1.0 if node_region in geographic_preferences else 0.5
            
            # Calculate overall compatibility
            overall_compatibility = (area_match * 0.4 + expertise_match * 0.4 + geographic_match * 0.2)
            
            if overall_compatibility > 0.5:
                collaborator_info = {
                    "node_id": node_id,
                    "node_name": node.name,
                    "node_type": node.node_type.value,
                    "compatibility_score": overall_compatibility,
                    "area_match": area_match,
                    "expertise_match": expertise_match,
                    "geographic_match": geographic_match,
                    "location": node.location,
                    "reputation": node.network_reputation
                }
                matching_collaborators.append(collaborator_info)
        
        # Sort by compatibility score
        matching_collaborators.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        return matching_collaborators
    
    async def _calculate_collaboration_score(self, node_a: ResearchNode, node_b: ResearchNode) -> float:
        """Calculate collaboration score between two nodes."""
        # Research area overlap
        area_overlap = len(set(node_a.research_areas) & set(node_b.research_areas))
        area_score = area_overlap / max(len(node_a.research_areas), len(node_b.research_areas), 1)
        
        # Complementary expertise
        expertise_complementarity = 0.0
        all_areas = set(node_a.research_areas) | set(node_b.research_areas)
        for area in all_areas:
            expertise_a = node_a.expertise_level.get(area, 0.0)
            expertise_b = node_b.expertise_level.get(area, 0.0)
            # Higher score for complementary expertise levels
            expertise_complementarity += min(expertise_a, expertise_b) + 0.5 * abs(expertise_a - expertise_b)
        expertise_complementarity /= max(len(all_areas), 1)
        
        # Reputation compatibility
        reputation_score = min(node_a.network_reputation, node_b.network_reputation)
        
        # Geographic accessibility (inverse of distance, normalized)
        distance = self._calculate_distance(node_a.location, node_b.location)
        geographic_score = 1.0 / (1.0 + distance / 10000)  # Normalize by 10,000km
        
        # Weighted combination
        collaboration_score = (
            area_score * 0.3 +
            expertise_complementarity * 0.3 +
            reputation_score * 0.2 +
            geographic_score * 0.2
        )
        
        return collaboration_score
    
    def _calculate_distance(self, loc_a: Tuple[float, float], loc_b: Tuple[float, float]) -> float:
        """Calculate distance between two geographic locations (simplified)."""
        lat1, lon1 = loc_a
        lat2, lon2 = loc_b
        
        # Simplified distance calculation (for more accuracy, use haversine formula)
        distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111  # Approximate km per degree
        return distance


class OpportunityDiscoverer:
    """Research opportunity discovery engine."""
    
    async def discover_opportunities(self,
                                   research_focus: List[str],
                                   nodes: Dict[str, ResearchNode],
                                   knowledge: Dict[str, KnowledgeElement]) -> List[ResearchOpportunity]:
        """Discover research opportunities."""
        opportunities = []
        
        # Funding opportunities (simulated)
        funding_opportunities = await self._discover_funding_opportunities(research_focus)
        opportunities.extend(funding_opportunities)
        
        # Collaboration opportunities
        collaboration_opportunities = await self._discover_collaboration_opportunities(research_focus, nodes)
        opportunities.extend(collaboration_opportunities)
        
        # Data sharing opportunities
        data_opportunities = await self._discover_data_opportunities(research_focus, knowledge)
        opportunities.extend(data_opportunities)
        
        # Infrastructure sharing opportunities
        infrastructure_opportunities = await self._discover_infrastructure_opportunities(research_focus, nodes)
        opportunities.extend(infrastructure_opportunities)
        
        return opportunities
    
    async def _discover_funding_opportunities(self, research_focus: List[str]) -> List[ResearchOpportunity]:
        """Discover funding opportunities."""
        # Simulated funding opportunities
        funding_sources = ["NSF", "NIH", "EU Horizon", "Industry Partners", "Private Foundations"]
        opportunities = []
        
        for area in research_focus:
            for source in funding_sources:
                opportunity = ResearchOpportunity(
                    opportunity_id=f"funding_{uuid.uuid4().hex[:8]}",
                    title=f"{source} Grant for {area} Research",
                    description=f"Funding opportunity in {area} from {source}",
                    research_areas=[area],
                    required_expertise={area: 0.7},
                    funding_available=np.random.uniform(50000, 2000000),  # Random funding amount
                    duration=timedelta(days=np.random.randint(365, 1095)),  # 1-3 years
                    priority=0.0,  # Will be calculated later
                    potential_participants=[],
                    application_deadline=datetime.now() + timedelta(days=np.random.randint(30, 180)),
                    opportunity_type="funding"
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    async def _discover_collaboration_opportunities(self,
                                                  research_focus: List[str],
                                                  nodes: Dict[str, ResearchNode]) -> List[ResearchOpportunity]:
        """Discover collaboration opportunities."""
        opportunities = []
        
        # Look for nodes with complementary expertise
        for area in research_focus:
            relevant_nodes = [
                node for node in nodes.values()
                if area in node.research_areas and node.expertise_level.get(area, 0.0) > 0.6
            ]
            
            if len(relevant_nodes) >= 2:  # Need at least 2 nodes for collaboration
                opportunity = ResearchOpportunity(
                    opportunity_id=f"collab_{uuid.uuid4().hex[:8]}",
                    title=f"Multi-institutional {area} Collaboration",
                    description=f"Opportunity for collaborative research in {area}",
                    research_areas=[area],
                    required_expertise={area: 0.6},
                    funding_available=0.0,  # Collaboration opportunity, not funding
                    duration=timedelta(days=730),  # 2 years
                    priority=0.0,
                    potential_participants=[node.node_id for node in relevant_nodes[:5]],
                    application_deadline=datetime.now() + timedelta(days=60),
                    opportunity_type="collaboration"
                )
                opportunities.append(opportunity)
        
        return opportunities


class KnowledgeSynthesizer:
    """Global knowledge synthesis engine."""
    
    async def find_related_knowledge(self,
                                   knowledge: KnowledgeElement,
                                   existing_knowledge: Dict[str, KnowledgeElement]) -> Dict[str, float]:
        """Find knowledge elements related to a given element."""
        related_knowledge = {}
        
        for element_id, element in existing_knowledge.items():
            if element_id == knowledge.knowledge_id:
                continue
            
            # Calculate relationship strength
            relationship_strength = await self._calculate_knowledge_relationship(knowledge, element)
            
            if relationship_strength > 0.3:  # Threshold for relationship
                related_knowledge[element_id] = relationship_strength
        
        return related_knowledge
    
    async def synthesize_global_knowledge(self,
                                        research_question: str,
                                        knowledge_elements: Dict[str, KnowledgeElement],
                                        nodes: Dict[str, ResearchNode]) -> Dict[str, Any]:
        """Synthesize knowledge from global network."""
        # Find relevant knowledge elements
        relevant_knowledge = []
        for knowledge in knowledge_elements.values():
            relevance = await self._calculate_question_relevance(research_question, knowledge)
            if relevance > 0.5:
                relevant_knowledge.append((knowledge, relevance))
        
        # Sort by relevance
        relevant_knowledge.sort(key=lambda x: x[1], reverse=True)
        
        # Synthesize knowledge
        synthesis = await self._perform_knowledge_synthesis([k[0] for k in relevant_knowledge])
        
        # Identify knowledge gaps
        gaps = await self._identify_knowledge_gaps(research_question, relevant_knowledge)
        
        # Generate recommendations
        recommendations = await self._generate_research_recommendations(synthesis, gaps)
        
        # Analyze cultural perspectives
        cultural_perspectives = await self._analyze_cultural_perspectives(relevant_knowledge, nodes)
        
        return {
            "synthesis": synthesis,
            "sources": [k.knowledge_id for k, _ in relevant_knowledge],
            "confidence": self._calculate_synthesis_confidence(relevant_knowledge),
            "gaps": gaps,
            "recommendations": recommendations,
            "cultural_perspectives": cultural_perspectives
        }
    
    async def identify_emerging_trends(self, knowledge_elements: Dict[str, KnowledgeElement]) -> List[IntelligenceInsight]:
        """Identify emerging research trends."""
        insights = []
        
        # Analyze recent knowledge elements for trends
        recent_knowledge = [
            k for k in knowledge_elements.values()
            if (datetime.now() - k.creation_date).days <= 180  # Last 6 months
        ]
        
        # Group by research areas
        area_knowledge = defaultdict(list)
        for knowledge in recent_knowledge:
            for area in knowledge.research_areas:
                area_knowledge[area].append(knowledge)
        
        # Identify trends in each area
        for area, area_elements in area_knowledge.items():
            if len(area_elements) >= 5:  # Need sufficient data
                # Analyze growth trend
                monthly_counts = defaultdict(int)
                for element in area_elements:
                    month_key = element.creation_date.strftime("%Y-%m")
                    monthly_counts[month_key] += 1
                
                # Check for increasing trend
                counts = list(monthly_counts.values())
                if len(counts) >= 3:
                    recent_growth = sum(counts[-2:]) / sum(counts[:-2]) if sum(counts[:-2]) > 0 else float('inf')
                    
                    if recent_growth > 1.5:  # 50% growth threshold
                        insight = IntelligenceInsight(
                            insight_id=f"trend_{uuid.uuid4().hex[:8]}",
                            title=f"Emerging Trend in {area}",
                            description=f"Increasing research activity detected in {area} with {recent_growth:.1f}x growth",
                            insight_type="emerging_trend",
                            confidence=min(0.9, recent_growth / 3.0),
                            affected_nodes=[],  # Would be populated with relevant researchers
                            supporting_evidence=[f"Recent publications: {len(area_elements)}"],
                            recommendations=[f"Consider increased focus on {area} research"],
                            generated_at=datetime.now(),
                            expiration_date=datetime.now() + timedelta(days=90),
                            impact_score=min(1.0, recent_growth / 2.0)
                        )
                        insights.append(insight)
        
        return insights
    
    async def _calculate_knowledge_relationship(self,
                                              knowledge_a: KnowledgeElement,
                                              knowledge_b: KnowledgeElement) -> float:
        """Calculate relationship strength between knowledge elements."""
        # Research area overlap
        area_overlap = len(set(knowledge_a.research_areas) & set(knowledge_b.research_areas))
        area_score = area_overlap / max(len(knowledge_a.research_areas), len(knowledge_b.research_areas), 1)
        
        # Author overlap
        author_overlap = len(set(knowledge_a.authors) & set(knowledge_b.authors))
        author_score = author_overlap / max(len(knowledge_a.authors), len(knowledge_b.authors), 1)
        
        # Content similarity (simplified - would use NLP in practice)
        content_similarity = self._calculate_content_similarity(knowledge_a.content, knowledge_b.content)
        
        # Citation relationship (if one cites the other)
        citation_score = 0.2 if knowledge_b.knowledge_id in knowledge_a.linked_elements else 0.0
        
        # Temporal relationship (recent papers are more related)
        time_diff = abs((knowledge_a.creation_date - knowledge_b.creation_date).days)
        temporal_score = max(0.0, 1.0 - time_diff / 1095)  # Decay over 3 years
        
        # Weighted combination
        relationship_strength = (
            area_score * 0.3 +
            content_similarity * 0.25 +
            author_score * 0.2 +
            citation_score * 0.15 +
            temporal_score * 0.1
        )
        
        return relationship_strength
    
    def _calculate_content_similarity(self, content_a: str, content_b: str) -> float:
        """Calculate content similarity (simplified)."""
        # Convert to word sets
        words_a = set(content_a.lower().split())
        words_b = set(content_b.lower().split())
        
        # Calculate Jaccard similarity
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union


class NetworkAnalyzer:
    """Network structure and dynamics analyzer."""
    
    async def analyze_collaboration_patterns(self,
                                           collaboration_graph: nx.Graph,
                                           nodes: Dict[str, ResearchNode]) -> List[IntelligenceInsight]:
        """Analyze collaboration patterns in the network."""
        insights = []
        
        try:
            # Identify highly connected nodes (hubs)
            centrality = nx.degree_centrality(collaboration_graph)
            hubs = [node_id for node_id, centrality_score in centrality.items() if centrality_score > 0.1]
            
            if hubs:
                insight = IntelligenceInsight(
                    insight_id=f"hubs_{uuid.uuid4().hex[:8]}",
                    title="Research Collaboration Hubs Identified",
                    description=f"Identified {len(hubs)} highly connected research hubs in the network",
                    insight_type="network_structure",
                    confidence=0.8,
                    affected_nodes=hubs,
                    supporting_evidence=[f"Hub centrality scores: {[centrality[h] for h in hubs]}"],
                    recommendations=["Consider leveraging hubs for network-wide initiatives"],
                    generated_at=datetime.now(),
                    expiration_date=datetime.now() + timedelta(days=180),
                    impact_score=0.7
                )
                insights.append(insight)
            
            # Identify communities
            try:
                communities = nx.community.greedy_modularity_communities(collaboration_graph)
                if len(communities) > 1:
                    insight = IntelligenceInsight(
                        insight_id=f"communities_{uuid.uuid4().hex[:8]}",
                        title="Research Communities Detected",
                        description=f"Detected {len(communities)} distinct research communities",
                        insight_type="community_structure",
                        confidence=0.7,
                        affected_nodes=[],
                        supporting_evidence=[f"Community sizes: {[len(c) for c in communities]}"],
                        recommendations=["Foster inter-community collaboration"],
                        generated_at=datetime.now(),
                        expiration_date=datetime.now() + timedelta(days=120),
                        impact_score=0.6
                    )
                    insights.append(insight)
            except:
                pass  # Community detection may fail for small networks
            
        except Exception as e:
            logger.error(f"Error analyzing collaboration patterns: {e}")
        
        return insights
    
    async def identify_network_bottlenecks(self, collaboration_graph: nx.Graph) -> List[IntelligenceInsight]:
        """Identify bottlenecks in the collaboration network."""
        insights = []
        
        try:
            if collaboration_graph.number_of_nodes() > 3:
                # Identify bridge nodes (articulation points)
                articulation_points = list(nx.articulation_points(collaboration_graph))
                
                if articulation_points:
                    insight = IntelligenceInsight(
                        insight_id=f"bottlenecks_{uuid.uuid4().hex[:8]}",
                        title="Network Bottlenecks Identified",
                        description=f"Identified {len(articulation_points)} critical bottleneck nodes",
                        insight_type="network_bottleneck",
                        confidence=0.9,
                        affected_nodes=articulation_points,
                        supporting_evidence=["Removal of these nodes would disconnect network components"],
                        recommendations=[
                            "Strengthen connections around bottleneck nodes",
                            "Create alternative pathways to reduce dependency"
                        ],
                        generated_at=datetime.now(),
                        expiration_date=datetime.now() + timedelta(days=90),
                        impact_score=0.8
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error identifying network bottlenecks: {e}")
        
        return insights


class ReputationEngine:
    """Network reputation management system."""
    
    async def initialize_node_reputation(self, node: ResearchNode) -> float:
        """Initialize reputation score for a new node."""
        # Base reputation based on node type
        base_scores = {
            NetworkNodeType.RESEARCH_INSTITUTION: 0.7,
            NetworkNodeType.INDIVIDUAL_RESEARCHER: 0.6,
            NetworkNodeType.INDUSTRY_PARTNER: 0.6,
            NetworkNodeType.FUNDING_AGENCY: 0.8,
            NetworkNodeType.AI_ASSISTANT: 0.5
        }
        
        base_reputation = base_scores.get(node.node_type, 0.5)
        
        # Adjust based on expertise levels
        avg_expertise = statistics.mean(node.expertise_level.values()) if node.expertise_level else 0.5
        expertise_bonus = (avg_expertise - 0.5) * 0.2
        
        initial_reputation = min(1.0, max(0.0, base_reputation + expertise_bonus))
        node.network_reputation = initial_reputation
        
        return initial_reputation
    
    async def update_collaboration_reputation(self, collaboration_link: CollaborationLink) -> None:
        """Update reputation scores based on collaboration outcomes."""
        # This would implement reputation updates based on collaboration success
        # For now, it's a placeholder
        pass


class DistributedAICoordinator:
    """Coordinator for distributed AI tasks across network."""
    
    async def coordinate_distributed_task(self,
                                         task: Dict[str, Any],
                                         participating_nodes: List[str],
                                         nodes: Dict[str, ResearchNode]) -> Dict[str, Any]:
        """Coordinate distributed AI computation."""
        coordination_result = {
            "task_id": task.get("task_id", "unknown"),
            "participating_nodes": participating_nodes,
            "node_results": {},
            "aggregated_result": None,
            "efficiency": 0.0,
            "coordination_time": 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Simulate distributed computation
            for node_id in participating_nodes:
                if node_id in nodes:
                    # Simulate node computation
                    node_result = await self._simulate_node_computation(task, nodes[node_id])
                    coordination_result["node_results"][node_id] = node_result
            
            # Aggregate results
            if coordination_result["node_results"]:
                coordination_result["aggregated_result"] = await self._aggregate_node_results(
                    coordination_result["node_results"]
                )
            
            # Calculate efficiency
            end_time = datetime.now()
            coordination_result["coordination_time"] = (end_time - start_time).total_seconds()
            coordination_result["efficiency"] = len(coordination_result["node_results"]) / max(len(participating_nodes), 1)
            
        except Exception as e:
            logger.error(f"Distributed AI coordination failed: {e}")
            coordination_result["error"] = str(e)
        
        return coordination_result
    
    async def _simulate_node_computation(self, task: Dict[str, Any], node: ResearchNode) -> Dict[str, Any]:
        """Simulate computation at a network node."""
        # Simulate processing time based on node capacity
        processing_time = 1.0 / max(node.collaboration_capacity, 1)
        await asyncio.sleep(min(processing_time, 0.1))  # Cap simulation time
        
        # Simulate result based on node expertise
        task_area = task.get("research_area", "general")
        node_expertise = node.expertise_level.get(task_area, 0.5)
        
        return {
            "node_id": node.node_id,
            "computation_result": node_expertise * np.random.uniform(0.8, 1.2),  # Simulate result
            "processing_time": processing_time,
            "confidence": node_expertise
        }
    
    async def _aggregate_node_results(self, node_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple nodes."""
        if not node_results:
            return {}
        
        # Weighted average based on confidence
        total_weighted_result = 0.0
        total_weight = 0.0
        
        for result in node_results.values():
            weight = result.get("confidence", 0.5)
            value = result.get("computation_result", 0.0)
            total_weighted_result += value * weight
            total_weight += weight
        
        aggregated_value = total_weighted_result / max(total_weight, 0.001)
        
        return {
            "aggregated_computation": aggregated_value,
            "contributing_nodes": len(node_results),
            "average_confidence": total_weight / len(node_results)
        }


class ConsensusEngine:
    """Consensus mechanism for distributed decisions."""
    
    async def achieve_consensus(self, 
                              node_results: Dict[str, Dict[str, Any]],
                              participating_nodes: List[str]) -> Dict[str, Any]:
        """Achieve consensus on distributed computation results."""
        consensus_result = {
            "consensus_achieved": False,
            "consensus_value": None,
            "consensus_confidence": 0.0,
            "dissenting_nodes": [],
            "consensus_method": "weighted_average"
        }
        
        try:
            if not node_results:
                return consensus_result
            
            # Extract computation results and confidences
            results = []
            confidences = []
            
            for node_result in node_results.values():
                if "computation_result" in node_result and "confidence" in node_result:
                    results.append(node_result["computation_result"])
                    confidences.append(node_result["confidence"])
            
            if results:
                # Calculate consensus using weighted average
                weights = np.array(confidences)
                values = np.array(results)
                
                consensus_value = np.average(values, weights=weights)
                consensus_confidence = np.mean(confidences)
                
                # Check for consensus (low variance indicates agreement)
                if len(results) > 1:
                    variance = np.var(results)
                    consensus_threshold = 0.1  # Threshold for consensus
                    consensus_achieved = variance < consensus_threshold
                else:
                    consensus_achieved = True
                
                consensus_result.update({
                    "consensus_achieved": consensus_achieved,
                    "consensus_value": float(consensus_value),
                    "consensus_confidence": float(consensus_confidence),
                    "result_variance": float(variance) if len(results) > 1 else 0.0
                })
            
        except Exception as e:
            logger.error(f"Consensus calculation failed: {e}")
            consensus_result["error"] = str(e)
        
        return consensus_result
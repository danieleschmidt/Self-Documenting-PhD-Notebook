"""
Advanced Research Collaboration Engine
Facilitates intelligent academic networking, collaboration discovery, and project management.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import networkx as nx
from collections import defaultdict, Counter

from ..core.note import Note, NoteType
from ..utils.logging import setup_logger


class CollaborationType(Enum):
    COAUTHOR = "coauthor"
    ADVISOR = "advisor"
    MENTOR = "mentor"
    PEER_REVIEWER = "peer_reviewer"
    RESEARCH_PARTNER = "research_partner"
    DOMAIN_EXPERT = "domain_expert"
    DATA_PROVIDER = "data_provider"
    TECHNICAL_EXPERT = "technical_expert"


class CollaborationStatus(Enum):
    POTENTIAL = "potential"
    INITIATED = "initiated"
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    DECLINED = "declined"


class InteractionType(Enum):
    EMAIL = "email"
    MEETING = "meeting"
    PAPER_REVIEW = "paper_review"
    CONFERENCE = "conference"
    WORKSHOP = "workshop"
    INFORMAL = "informal"
    PUBLICATION = "publication"


@dataclass
class Collaborator:
    """Academic collaborator profile."""
    id: str
    name: str
    email: Optional[str]
    institution: str
    department: Optional[str]
    position: str  # PhD Student, Postdoc, Professor, etc.
    
    # Research profile
    research_interests: List[str]
    expertise_areas: List[str]
    publications: List[str]  # DOIs or titles
    h_index: Optional[int]
    
    # Collaboration metrics
    collaboration_strength: float = 0.0  # 0-1 based on interaction history
    responsiveness_score: float = 0.5  # How quickly they respond
    collaboration_quality: float = 0.5  # Quality of past collaborations
    
    # Contact info
    website: Optional[str] = None
    social_profiles: Dict[str, str] = None
    preferred_contact: str = "email"
    timezone: Optional[str] = None
    
    # Metadata
    first_contact: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    notes: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.social_profiles is None:
            self.social_profiles = {}
        if self.tags is None:
            self.tags = []
            
    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.first_contact:
            result['first_contact'] = self.first_contact.isoformat()
        if self.last_interaction:
            result['last_interaction'] = self.last_interaction.isoformat()
        return result


@dataclass
class Collaboration:
    """A specific collaboration instance."""
    id: str
    title: str
    description: str
    collaborators: List[str]  # Collaborator IDs
    lead_researcher: str  # Collaborator ID
    
    # Collaboration details
    collaboration_type: CollaborationType
    status: CollaborationStatus
    start_date: datetime
    expected_end_date: Optional[datetime]
    actual_end_date: Optional[datetime]
    
    # Project specifics
    research_goals: List[str]
    deliverables: List[str]
    milestones: List[Dict[str, Any]]
    resources_shared: List[str]
    
    # Communication
    meeting_frequency: Optional[str] = None  # "weekly", "monthly", etc.
    communication_channels: List[str] = None
    shared_documents: List[str] = None
    
    # Outcomes
    publications: List[str] = None
    presentations: List[str] = None
    outcomes: List[str] = None
    success_metrics: Dict[str, Any] = None
    
    # Metadata
    created_at: datetime = None
    updated_at: Optional[datetime] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.communication_channels is None:
            self.communication_channels = []
        if self.shared_documents is None:
            self.shared_documents = []
        if self.publications is None:
            self.publications = []
        if self.presentations is None:
            self.presentations = []
        if self.outcomes is None:
            self.outcomes = []
        if self.success_metrics is None:
            self.success_metrics = {}
        if self.created_at is None:
            self.created_at = datetime.now()
            
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['collaboration_type'] = self.collaboration_type.value
        result['status'] = self.status.value
        result['start_date'] = self.start_date.isoformat()
        result['created_at'] = self.created_at.isoformat()
        
        if self.expected_end_date:
            result['expected_end_date'] = self.expected_end_date.isoformat()
        if self.actual_end_date:
            result['actual_end_date'] = self.actual_end_date.isoformat()
        if self.updated_at:
            result['updated_at'] = self.updated_at.isoformat()
            
        return result


@dataclass
class Interaction:
    """Record of an interaction with a collaborator."""
    id: str
    collaborator_id: str
    interaction_type: InteractionType
    date: datetime
    duration_minutes: Optional[int]
    
    title: str
    description: str
    outcomes: List[str]
    action_items: List[Dict[str, Any]]
    
    # Quality metrics
    productivity_score: float = 0.5  # How productive was this interaction
    satisfaction_score: float = 0.5  # How satisfactory was this interaction
    
    # Context
    collaboration_id: Optional[str] = None
    location: Optional[str] = None
    participants: List[str] = None
    attachments: List[str] = None
    
    # Follow-up
    follow_up_required: bool = False
    follow_up_date: Optional[datetime] = None
    follow_up_completed: bool = False
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = []
        if self.attachments is None:
            self.attachments = []
            
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['interaction_type'] = self.interaction_type.value
        result['date'] = self.date.isoformat()
        if self.follow_up_date:
            result['follow_up_date'] = self.follow_up_date.isoformat()
        return result


class CollaborationEngine:
    """
    Advanced engine for managing academic collaborations, networking,
    and research partnerships with intelligent recommendation systems.
    """
    
    def __init__(self, notebook_path: Path):
        self.logger = setup_logger("research.collaboration_engine")
        self.notebook_path = notebook_path
        
        # Data stores
        self.collaborators: Dict[str, Collaborator] = {}
        self.collaborations: Dict[str, Collaboration] = {}
        self.interactions: Dict[str, Interaction] = {}
        
        # Network analysis
        self.collaboration_network = nx.Graph()
        
        # Create directories
        self.collab_dir = notebook_path / "research" / "collaborations"
        self.profiles_dir = self.collab_dir / "profiles"
        self.projects_dir = self.collab_dir / "projects"
        self.interactions_dir = self.collab_dir / "interactions"
        
        for dir_path in [self.collab_dir, self.profiles_dir, self.projects_dir, self.interactions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing collaboration data."""
        try:
            # Load collaborators
            for profile_file in self.profiles_dir.glob("*.json"):
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    collaborator = self._dict_to_collaborator(data)
                    self.collaborators[collaborator.id] = collaborator
            
            # Load collaborations
            for proj_file in self.projects_dir.glob("*.json"):
                with open(proj_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    collaboration = self._dict_to_collaboration(data)
                    self.collaborations[collaboration.id] = collaboration
            
            # Load interactions
            for int_file in self.interactions_dir.glob("*.json"):
                with open(int_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    interaction = self._dict_to_interaction(data)
                    self.interactions[interaction.id] = interaction
            
            self._rebuild_network()
            
            self.logger.info(f"Loaded {len(self.collaborators)} collaborators, "
                           f"{len(self.collaborations)} collaborations, "
                           f"{len(self.interactions)} interactions")
                           
        except Exception as e:
            self.logger.error(f"Error loading collaboration data: {e}")
    
    def _dict_to_collaborator(self, data: Dict) -> Collaborator:
        """Convert dictionary to Collaborator object."""
        if data.get('first_contact'):
            data['first_contact'] = datetime.fromisoformat(data['first_contact'])
        if data.get('last_interaction'):
            data['last_interaction'] = datetime.fromisoformat(data['last_interaction'])
        return Collaborator(**data)
    
    def _dict_to_collaboration(self, data: Dict) -> Collaboration:
        """Convert dictionary to Collaboration object."""
        data['collaboration_type'] = CollaborationType(data['collaboration_type'])
        data['status'] = CollaborationStatus(data['status'])
        data['start_date'] = datetime.fromisoformat(data['start_date'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        if data.get('expected_end_date'):
            data['expected_end_date'] = datetime.fromisoformat(data['expected_end_date'])
        if data.get('actual_end_date'):
            data['actual_end_date'] = datetime.fromisoformat(data['actual_end_date'])
        if data.get('updated_at'):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
        return Collaboration(**data)
    
    def _dict_to_interaction(self, data: Dict) -> Interaction:
        """Convert dictionary to Interaction object."""
        data['interaction_type'] = InteractionType(data['interaction_type'])
        data['date'] = datetime.fromisoformat(data['date'])
        if data.get('follow_up_date'):
            data['follow_up_date'] = datetime.fromisoformat(data['follow_up_date'])
        return Interaction(**data)
    
    def add_collaborator(
        self,
        name: str,
        email: str,
        institution: str,
        position: str,
        research_interests: List[str],
        expertise_areas: List[str],
        **kwargs
    ) -> Collaborator:
        """Add a new collaborator to the network."""
        
        # Generate unique ID
        collaborator_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check for existing collaborator by email
        existing = self._find_collaborator_by_email(email)
        if existing:
            self.logger.warning(f"Collaborator with email {email} already exists")
            return existing
        
        collaborator = Collaborator(
            id=collaborator_id,
            name=name.strip(),
            email=email.strip().lower(),
            institution=institution.strip(),
            position=position.strip(),
            research_interests=research_interests,
            expertise_areas=expertise_areas,
            first_contact=datetime.now(),
            **kwargs
        )
        
        # Add to stores
        self.collaborators[collaborator_id] = collaborator
        self._save_collaborator(collaborator)
        
        # Create profile note
        self._create_collaborator_note(collaborator)
        
        # Update network
        self.collaboration_network.add_node(collaborator_id, **{
            'name': collaborator.name,
            'institution': collaborator.institution,
            'position': collaborator.position,
            'expertise': collaborator.expertise_areas
        })
        
        self.logger.info(f"Added collaborator: {collaborator_id} - {name}")
        return collaborator
    
    def _find_collaborator_by_email(self, email: str) -> Optional[Collaborator]:
        """Find collaborator by email address."""
        email_lower = email.lower().strip()
        for collaborator in self.collaborators.values():
            if collaborator.email and collaborator.email.lower() == email_lower:
                return collaborator
        return None
    
    def create_collaboration(
        self,
        title: str,
        description: str,
        collaborator_ids: List[str],
        collaboration_type: CollaborationType,
        lead_researcher_id: str,
        research_goals: List[str],
        **kwargs
    ) -> Collaboration:
        """Create a new collaboration project."""
        
        # Validate collaborators exist
        for collab_id in collaborator_ids:
            if collab_id not in self.collaborators:
                raise ValueError(f"Collaborator {collab_id} not found")
        
        if lead_researcher_id not in self.collaborators:
            raise ValueError(f"Lead researcher {lead_researcher_id} not found")
        
        # Generate unique ID
        collaboration_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        collaboration = Collaboration(
            id=collaboration_id,
            title=title.strip(),
            description=description.strip(),
            collaborators=collaborator_ids,
            lead_researcher=lead_researcher_id,
            collaboration_type=collaboration_type,
            status=CollaborationStatus.INITIATED,
            start_date=datetime.now(),
            research_goals=research_goals,
            deliverables=[],
            milestones=[],
            resources_shared=[],
            **kwargs
        )
        
        # Add to store
        self.collaborations[collaboration_id] = collaboration
        self._save_collaboration(collaboration)
        
        # Create project note
        self._create_collaboration_note(collaboration)
        
        # Update network edges
        for i, collab_id_1 in enumerate(collaborator_ids):
            for collab_id_2 in collaborator_ids[i+1:]:
                self._add_collaboration_edge(collab_id_1, collab_id_2, collaboration_id)
        
        self.logger.info(f"Created collaboration: {collaboration_id} - {title}")
        return collaboration
    
    def _add_collaboration_edge(self, collab1: str, collab2: str, collaboration_id: str):
        """Add or strengthen edge between collaborators in network."""
        if self.collaboration_network.has_edge(collab1, collab2):
            # Strengthen existing connection
            edge_data = self.collaboration_network[collab1][collab2]
            edge_data['weight'] = edge_data.get('weight', 1) + 1
            edge_data['collaborations'] = edge_data.get('collaborations', [])
            edge_data['collaborations'].append(collaboration_id)
        else:
            # Create new connection
            self.collaboration_network.add_edge(collab1, collab2, 
                                              weight=1, 
                                              collaborations=[collaboration_id])
    
    def record_interaction(
        self,
        collaborator_id: str,
        interaction_type: InteractionType,
        title: str,
        description: str,
        duration_minutes: Optional[int] = None,
        outcomes: List[str] = None,
        collaboration_id: Optional[str] = None,
        **kwargs
    ) -> Interaction:
        """Record an interaction with a collaborator."""
        
        if collaborator_id not in self.collaborators:
            raise ValueError(f"Collaborator {collaborator_id} not found")
        
        # Generate unique ID
        interaction_id = f"int_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        interaction = Interaction(
            id=interaction_id,
            collaborator_id=collaborator_id,
            interaction_type=interaction_type,
            date=datetime.now(),
            duration_minutes=duration_minutes,
            title=title.strip(),
            description=description.strip(),
            outcomes=outcomes or [],
            action_items=[],
            collaboration_id=collaboration_id,
            **kwargs
        )
        
        # Add to store
        self.interactions[interaction_id] = interaction
        self._save_interaction(interaction)
        
        # Update collaborator's last interaction
        self.collaborators[collaborator_id].last_interaction = datetime.now()
        self._update_collaborator_metrics(collaborator_id)
        self._save_collaborator(self.collaborators[collaborator_id])
        
        self.logger.info(f"Recorded interaction: {interaction_id}")
        return interaction
    
    def _update_collaborator_metrics(self, collaborator_id: str):
        """Update collaboration metrics for a collaborator."""
        collaborator = self.collaborators[collaborator_id]
        
        # Get all interactions with this collaborator
        collaborator_interactions = [
            interaction for interaction in self.interactions.values()
            if interaction.collaborator_id == collaborator_id
        ]
        
        if not collaborator_interactions:
            return
        
        # Calculate metrics
        total_interactions = len(collaborator_interactions)
        recent_interactions = [
            interaction for interaction in collaborator_interactions
            if (datetime.now() - interaction.date).days <= 90
        ]
        
        # Collaboration strength (based on frequency and recency)
        base_strength = min(total_interactions / 10.0, 1.0)
        recency_bonus = len(recent_interactions) / max(total_interactions, 1) * 0.3
        collaborator.collaboration_strength = min(base_strength + recency_bonus, 1.0)
        
        # Average satisfaction and productivity
        if collaborator_interactions:
            avg_satisfaction = sum(i.satisfaction_score for i in collaborator_interactions) / total_interactions
            avg_productivity = sum(i.productivity_score for i in collaborator_interactions) / total_interactions
            
            collaborator.collaboration_quality = (avg_satisfaction + avg_productivity) / 2
        
        # Responsiveness (simplified - based on interaction frequency)
        if len(recent_interactions) >= 2:
            intervals = []
            sorted_interactions = sorted(recent_interactions, key=lambda x: x.date)
            for i in range(1, len(sorted_interactions)):
                interval = (sorted_interactions[i].date - sorted_interactions[i-1].date).days
                intervals.append(interval)
            
            avg_interval = sum(intervals) / len(intervals)
            # Convert to responsiveness score (shorter intervals = higher responsiveness)
            collaborator.responsiveness_score = max(0.1, 1.0 - (avg_interval / 30.0))
    
    def find_collaboration_opportunities(
        self,
        research_interests: List[str],
        expertise_needed: List[str],
        collaboration_type: Optional[CollaborationType] = None,
        min_score: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Find potential collaboration opportunities."""
        
        opportunities = []
        
        for collaborator_id, collaborator in self.collaborators.items():
            score = self._calculate_collaboration_score(
                collaborator, research_interests, expertise_needed
            )
            
            if score >= min_score:
                # Get additional context
                shared_connections = self._get_shared_connections(collaborator_id)
                interaction_history = [
                    i for i in self.interactions.values()
                    if i.collaborator_id == collaborator_id
                ]
                
                opportunity = {
                    "collaborator_id": collaborator_id,
                    "collaborator": collaborator,
                    "match_score": score,
                    "match_reasons": self._explain_match(collaborator, research_interests, expertise_needed),
                    "shared_connections": shared_connections,
                    "interaction_history_count": len(interaction_history),
                    "last_interaction": collaborator.last_interaction,
                    "collaboration_strength": collaborator.collaboration_strength,
                    "recommended_approach": self._suggest_approach(collaborator, score)
                }
                
                opportunities.append(opportunity)
        
        # Sort by score
        opportunities.sort(key=lambda x: x["match_score"], reverse=True)
        
        return opportunities
    
    def _calculate_collaboration_score(
        self,
        collaborator: Collaborator,
        research_interests: List[str],
        expertise_needed: List[str]
    ) -> float:
        """Calculate collaboration potential score."""
        
        score = 0.0
        
        # Interest overlap (40% weight)
        interest_matches = len(set(research_interests) & set(collaborator.research_interests))
        max_possible_matches = min(len(research_interests), len(collaborator.research_interests))
        if max_possible_matches > 0:
            interest_score = interest_matches / max_possible_matches
            score += interest_score * 0.4
        
        # Expertise match (35% weight)
        expertise_matches = len(set(expertise_needed) & set(collaborator.expertise_areas))
        if expertise_needed and expertise_matches > 0:
            expertise_score = expertise_matches / len(expertise_needed)
            score += expertise_score * 0.35
        
        # Collaboration history (15% weight)
        score += collaborator.collaboration_strength * 0.15
        
        # Quality of past collaborations (10% weight)
        score += collaborator.collaboration_quality * 0.10
        
        return min(score, 1.0)
    
    def _explain_match(
        self,
        collaborator: Collaborator,
        research_interests: List[str],
        expertise_needed: List[str]
    ) -> List[str]:
        """Explain why this collaborator is a good match."""
        reasons = []
        
        # Interest overlaps
        shared_interests = list(set(research_interests) & set(collaborator.research_interests))
        if shared_interests:
            reasons.append(f"Shared research interests: {', '.join(shared_interests)}")
        
        # Expertise matches
        matching_expertise = list(set(expertise_needed) & set(collaborator.expertise_areas))
        if matching_expertise:
            reasons.append(f"Needed expertise: {', '.join(matching_expertise)}")
        
        # Collaboration history
        if collaborator.collaboration_strength > 0.5:
            reasons.append("Strong previous collaboration history")
        
        # Institutional diversity
        # This would need to be compared against the user's institution
        reasons.append(f"Institutional perspective from {collaborator.institution}")
        
        return reasons
    
    def _suggest_approach(self, collaborator: Collaborator, score: float) -> str:
        """Suggest how to approach this collaborator."""
        
        if collaborator.last_interaction and (datetime.now() - collaborator.last_interaction).days < 30:
            return "Recent interaction - follow up on previous conversation"
        elif collaborator.collaboration_strength > 0.7:
            return "Strong relationship - direct collaboration proposal"
        elif score > 0.8:
            return "High match - direct outreach with specific proposal"
        elif collaborator.responsiveness_score > 0.7:
            return "Responsive collaborator - email introduction with research overview"
        else:
            return "Warm introduction through mutual connections recommended"
    
    def _get_shared_connections(self, collaborator_id: str) -> List[str]:
        """Get shared connections with a collaborator."""
        if not self.collaboration_network.has_node(collaborator_id):
            return []
        
        # This is simplified - in practice would analyze the network more thoroughly
        neighbors = list(self.collaboration_network.neighbors(collaborator_id))
        return neighbors[:5]  # Return top 5 connections
    
    def _rebuild_network(self):
        """Rebuild the collaboration network from stored data."""
        self.collaboration_network.clear()
        
        # Add all collaborators as nodes
        for collaborator_id, collaborator in self.collaborators.items():
            self.collaboration_network.add_node(collaborator_id, **{
                'name': collaborator.name,
                'institution': collaborator.institution,
                'position': collaborator.position,
                'expertise': collaborator.expertise_areas
            })
        
        # Add edges based on collaborations
        for collaboration in self.collaborations.values():
            collaborator_ids = collaboration.collaborators
            for i, collab_id_1 in enumerate(collaborator_ids):
                for collab_id_2 in collaborator_ids[i+1:]:
                    self._add_collaboration_edge(collab_id_1, collab_id_2, collaboration.id)
    
    def analyze_collaboration_network(self) -> Dict[str, Any]:
        """Analyze the collaboration network structure."""
        if not self.collaboration_network.nodes():
            return {"error": "No collaboration network data available"}
        
        analysis = {
            "network_size": {
                "nodes": self.collaboration_network.number_of_nodes(),
                "edges": self.collaboration_network.number_of_edges(),
                "density": nx.density(self.collaboration_network)
            },
            "centrality_analysis": {},
            "clustering": {},
            "components": {},
            "institution_analysis": {},
            "collaboration_patterns": {}
        }
        
        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.collaboration_network)
            closeness = nx.closeness_centrality(self.collaboration_network)
            degree = nx.degree_centrality(self.collaboration_network)
            
            # Top collaborators by different centrality measures
            analysis["centrality_analysis"] = {
                "most_connected": sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5],
                "key_bridges": sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5],
                "most_central": sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        except Exception as e:
            self.logger.warning(f"Error calculating centrality: {e}")
        
        # Clustering analysis
        try:
            clustering_coeff = nx.average_clustering(self.collaboration_network)
            analysis["clustering"]["average_clustering"] = clustering_coeff
            
            # Community detection (simplified)
            if self.collaboration_network.number_of_edges() > 0:
                communities = list(nx.connected_components(self.collaboration_network))
                analysis["clustering"]["communities"] = len(communities)
                analysis["clustering"]["largest_community"] = max(len(c) for c in communities)
        except Exception as e:
            self.logger.warning(f"Error calculating clustering: {e}")
        
        # Institutional analysis
        institutions = defaultdict(int)
        positions = defaultdict(int)
        
        for node_id in self.collaboration_network.nodes():
            if node_id in self.collaborators:
                collaborator = self.collaborators[node_id]
                institutions[collaborator.institution] += 1
                positions[collaborator.position] += 1
        
        analysis["institution_analysis"] = {
            "institution_distribution": dict(institutions),
            "position_distribution": dict(positions),
            "institutional_diversity": len(institutions)
        }
        
        # Collaboration patterns
        collaboration_types = Counter()
        active_collaborations = 0
        
        for collaboration in self.collaborations.values():
            collaboration_types[collaboration.collaboration_type.value] += 1
            if collaboration.status == CollaborationStatus.ACTIVE:
                active_collaborations += 1
        
        analysis["collaboration_patterns"] = {
            "types_distribution": dict(collaboration_types),
            "active_collaborations": active_collaborations,
            "total_collaborations": len(self.collaborations)
        }
        
        return analysis
    
    def get_collaboration_recommendations(
        self,
        collaborator_id: str,
        recommendation_type: str = "all"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get personalized collaboration recommendations."""
        
        if collaborator_id not in self.collaborators:
            raise ValueError(f"Collaborator {collaborator_id} not found")
        
        collaborator = self.collaborators[collaborator_id]
        recommendations = {
            "research_opportunities": [],
            "networking_suggestions": [],
            "project_ideas": [],
            "skill_development": []
        }
        
        # Research opportunity recommendations
        if recommendation_type in ["all", "research"]:
            # Find complementary researchers
            opportunities = self.find_collaboration_opportunities(
                research_interests=collaborator.research_interests,
                expertise_needed=collaborator.expertise_areas,
                min_score=0.4
            )
            
            recommendations["research_opportunities"] = opportunities[:5]
        
        # Networking suggestions
        if recommendation_type in ["all", "networking"]:
            network_suggestions = self._generate_networking_suggestions(collaborator_id)
            recommendations["networking_suggestions"] = network_suggestions
        
        # Project ideas based on expertise gaps
        if recommendation_type in ["all", "projects"]:
            project_ideas = self._suggest_project_ideas(collaborator)
            recommendations["project_ideas"] = project_ideas
        
        return recommendations
    
    def _generate_networking_suggestions(self, collaborator_id: str) -> List[Dict[str, Any]]:
        """Generate networking suggestions based on network analysis."""
        suggestions = []
        
        if not self.collaboration_network.has_node(collaborator_id):
            return suggestions
        
        # Suggest connections through mutual collaborators
        neighbors = list(self.collaboration_network.neighbors(collaborator_id))
        
        for neighbor in neighbors[:3]:  # Top 3 direct connections
            neighbor_neighbors = list(self.collaboration_network.neighbors(neighbor))
            
            for potential_connection in neighbor_neighbors:
                if (potential_connection != collaborator_id and 
                    not self.collaboration_network.has_edge(collaborator_id, potential_connection)):
                    
                    suggestions.append({
                        "type": "mutual_connection",
                        "collaborator_id": potential_connection,
                        "collaborator": self.collaborators.get(potential_connection),
                        "mutual_connection": self.collaborators.get(neighbor),
                        "reason": f"Connected through {self.collaborators[neighbor].name}"
                    })
        
        return suggestions[:5]  # Top 5 suggestions
    
    def _suggest_project_ideas(self, collaborator: Collaborator) -> List[Dict[str, Any]]:
        """Suggest project ideas based on collaborator's profile."""
        ideas = []
        
        # Cross-domain projects
        if len(collaborator.expertise_areas) > 1:
            for i, area1 in enumerate(collaborator.expertise_areas):
                for area2 in collaborator.expertise_areas[i+1:]:
                    ideas.append({
                        "type": "cross_domain",
                        "title": f"Interdisciplinary {area1}-{area2} Research",
                        "description": f"Explore connections between {area1} and {area2}",
                        "required_expertise": [area1, area2],
                        "potential_collaborators": self._find_experts([area1, area2], exclude_id=collaborator.id)
                    })
        
        # Trending research areas (simplified)
        trending_topics = ["artificial intelligence", "sustainability", "digital health", "quantum computing"]
        for interest in collaborator.research_interests:
            for topic in trending_topics:
                if topic.lower() not in interest.lower():
                    ideas.append({
                        "type": "trending_integration",
                        "title": f"{interest} meets {topic.title()}",
                        "description": f"Integrate {topic} approaches into {interest} research",
                        "required_expertise": [interest, topic],
                        "novelty_score": 0.8
                    })
        
        return ideas[:3]  # Top 3 suggestions
    
    def _find_experts(self, expertise_areas: List[str], exclude_id: str = None) -> List[str]:
        """Find collaborators with specific expertise areas."""
        experts = []
        
        for collaborator_id, collaborator in self.collaborators.items():
            if collaborator_id == exclude_id:
                continue
                
            if any(area.lower() in [exp.lower() for exp in collaborator.expertise_areas] 
                   for area in expertise_areas):
                experts.append(collaborator_id)
        
        return experts[:5]  # Top 5 experts
    
    def generate_collaboration_report(
        self,
        time_period_days: int = 365,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive collaboration analytics report."""
        
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        
        # Filter data by time period
        recent_collaborations = [
            c for c in self.collaborations.values()
            if c.start_date >= cutoff_date
        ]
        
        recent_interactions = [
            i for i in self.interactions.values()
            if i.date >= cutoff_date
        ]
        
        report = {
            "time_period": f"Last {time_period_days} days",
            "summary": {
                "total_collaborators": len(self.collaborators),
                "active_collaborations": len([c for c in recent_collaborations if c.status == CollaborationStatus.ACTIVE]),
                "new_collaborations": len(recent_collaborations),
                "total_interactions": len(recent_interactions),
                "avg_interactions_per_week": len(recent_interactions) / (time_period_days / 7)
            },
            "collaboration_analysis": self._analyze_collaboration_patterns(recent_collaborations),
            "interaction_analysis": self._analyze_interaction_patterns(recent_interactions),
            "network_analysis": self.analyze_collaboration_network(),
            "productivity_metrics": self._calculate_productivity_metrics(recent_collaborations, recent_interactions),
            "recommendations": self._generate_strategic_recommendations()
        }
        
        if include_predictions:
            report["predictions"] = self._generate_collaboration_predictions()
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.collab_dir / f"collaboration_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _analyze_collaboration_patterns(self, collaborations: List[Collaboration]) -> Dict[str, Any]:
        """Analyze patterns in collaborations."""
        if not collaborations:
            return {"error": "No collaborations to analyze"}
        
        # Type distribution
        type_counts = Counter(c.collaboration_type.value for c in collaborations)
        
        # Success rate
        completed = len([c for c in collaborations if c.status == CollaborationStatus.COMPLETED])
        success_rate = completed / len(collaborations) if collaborations else 0
        
        # Duration analysis
        durations = []
        for collab in collaborations:
            if collab.actual_end_date:
                duration = (collab.actual_end_date - collab.start_date).days
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "type_distribution": dict(type_counts),
            "success_rate": success_rate,
            "average_duration_days": avg_duration,
            "most_common_type": type_counts.most_common(1)[0] if type_counts else None
        }
    
    def _analyze_interaction_patterns(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """Analyze patterns in interactions."""
        if not interactions:
            return {"error": "No interactions to analyze"}
        
        # Type distribution
        type_counts = Counter(i.interaction_type.value for i in interactions)
        
        # Productivity analysis
        avg_productivity = sum(i.productivity_score for i in interactions) / len(interactions)
        avg_satisfaction = sum(i.satisfaction_score for i in interactions) / len(interactions)
        
        # Time analysis
        by_hour = defaultdict(int)
        by_day = defaultdict(int)
        
        for interaction in interactions:
            by_hour[interaction.date.hour] += 1
            by_day[interaction.date.strftime('%A')] += 1
        
        return {
            "type_distribution": dict(type_counts),
            "quality_metrics": {
                "avg_productivity": avg_productivity,
                "avg_satisfaction": avg_satisfaction
            },
            "time_patterns": {
                "peak_hours": sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:3],
                "active_days": dict(by_day)
            }
        }
    
    def _calculate_productivity_metrics(
        self,
        collaborations: List[Collaboration],
        interactions: List[Interaction]
    ) -> Dict[str, Any]:
        """Calculate collaboration productivity metrics."""
        
        # Publications per collaboration
        total_publications = sum(len(c.publications) for c in collaborations)
        publications_per_collab = total_publications / len(collaborations) if collaborations else 0
        
        # Outcome achievement rate
        total_outcomes = sum(len(c.outcomes) for c in collaborations)
        outcome_rate = total_outcomes / len(collaborations) if collaborations else 0
        
        # Interaction efficiency
        productive_interactions = len([i for i in interactions if i.productivity_score > 0.6])
        interaction_efficiency = productive_interactions / len(interactions) if interactions else 0
        
        return {
            "publications_per_collaboration": publications_per_collab,
            "outcome_achievement_rate": outcome_rate,
            "interaction_efficiency": interaction_efficiency,
            "total_outcomes_achieved": total_outcomes
        }
    
    def _generate_strategic_recommendations(self) -> List[Dict[str, str]]:
        """Generate strategic recommendations for collaboration improvement."""
        recommendations = []
        
        # Network analysis recommendations
        network_analysis = self.analyze_collaboration_network()
        
        if network_analysis.get("network_size", {}).get("density", 0) < 0.3:
            recommendations.append({
                "type": "network_expansion",
                "priority": "high",
                "recommendation": "Increase network density by introducing collaborators to each other",
                "action": "Organize networking events or joint meetings"
            })
        
        # Collaboration diversity
        active_types = set(c.collaboration_type.value for c in self.collaborations.values() 
                          if c.status == CollaborationStatus.ACTIVE)
        
        if len(active_types) < 3:
            recommendations.append({
                "type": "diversification",
                "priority": "medium",
                "recommendation": "Diversify collaboration types for broader impact",
                "action": "Explore mentoring, peer review, or technical expert collaborations"
            })
        
        # Institutional diversity
        institutions = set(c.institution for c in self.collaborators.values())
        if len(institutions) < 5:
            recommendations.append({
                "type": "institutional_diversity",
                "priority": "medium",
                "recommendation": "Expand institutional network for diverse perspectives",
                "action": "Reach out to researchers at different institutions"
            })
        
        return recommendations
    
    def _generate_collaboration_predictions(self) -> Dict[str, Any]:
        """Generate predictions about future collaboration trends."""
        # Simplified prediction model
        predictions = {}
        
        # Growth prediction
        recent_growth = len([c for c in self.collaborations.values() 
                           if (datetime.now() - c.created_at).days <= 90])
        
        predictions["network_growth"] = {
            "quarterly_new_collaborators": max(recent_growth, 1),
            "projected_network_size_next_year": len(self.collaborators) + (recent_growth * 4)
        }
        
        # Success prediction
        successful_collabs = len([c for c in self.collaborations.values() 
                                if c.status == CollaborationStatus.COMPLETED])
        success_rate = successful_collabs / max(len(self.collaborations), 1)
        
        predictions["success_trends"] = {
            "expected_success_rate": min(success_rate * 1.1, 0.95),  # Slight improvement
            "risk_factors": ["Limited follow-up", "Resource constraints", "Time management"]
        }
        
        return predictions
    
    def _create_collaborator_note(self, collaborator: Collaborator):
        """Create an Obsidian note for a collaborator."""
        note_path = self.profiles_dir / f"{collaborator.name.replace(' ', '_')}_{collaborator.id}.md"
        
        content = f"""# {collaborator.name}

## Contact Information
- **Email**: {collaborator.email}
- **Institution**: {collaborator.institution}
- **Department**: {collaborator.department or 'N/A'}
- **Position**: {collaborator.position}
- **Website**: {collaborator.website or 'N/A'}

## Research Profile
- **Research Interests**: {', '.join(collaborator.research_interests)}
- **Expertise Areas**: {', '.join(collaborator.expertise_areas)}
- **H-Index**: {collaborator.h_index or 'Unknown'}

## Collaboration Metrics
- **Collaboration Strength**: {collaborator.collaboration_strength:.1%}
- **Responsiveness Score**: {collaborator.responsiveness_score:.1%}
- **Collaboration Quality**: {collaborator.collaboration_quality:.1%}

## Key Publications
{"".join(f"- {pub}\\n" for pub in collaborator.publications)}

## Collaboration History
[This section will be updated with interaction history]

## Notes
{collaborator.notes or 'Add your notes about this collaborator here...'}

## Tags
{' '.join(f'#{tag}' for tag in collaborator.tags)}

---
*First Contact*: {collaborator.first_contact.strftime('%Y-%m-%d') if collaborator.first_contact else 'Unknown'}  
*Last Interaction*: {collaborator.last_interaction.strftime('%Y-%m-%d') if collaborator.last_interaction else 'None'}  
*Profile ID*: {collaborator.id}
"""
        
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_collaboration_note(self, collaboration: Collaboration):
        """Create an Obsidian note for a collaboration project."""
        note_path = self.projects_dir / f"{collaboration.title.replace(' ', '_')}_{collaboration.id}.md"
        
        # Get collaborator names
        collaborator_names = []
        for collab_id in collaboration.collaborators:
            if collab_id in self.collaborators:
                collaborator_names.append(self.collaborators[collab_id].name)
        
        content = f"""# {collaboration.title}

## Project Overview
{collaboration.description}

## Collaboration Details
- **Type**: {collaboration.collaboration_type.value}
- **Status**: {collaboration.status.value}
- **Lead Researcher**: {self.collaborators.get(collaboration.lead_researcher, {}).get('name', 'Unknown')}
- **Start Date**: {collaboration.start_date.strftime('%Y-%m-%d')}
- **Expected End**: {collaboration.expected_end_date.strftime('%Y-%m-%d') if collaboration.expected_end_date else 'TBD'}

## Collaborators
{"".join(f"- [[{name}]]\\n" for name in collaborator_names)}

## Research Goals
{"".join(f"- {goal}\\n" for goal in collaboration.research_goals)}

## Deliverables
{"".join(f"- {deliverable}\\n" for deliverable in collaboration.deliverables)}

## Milestones
{"".join(f"- {milestone}\\n" for milestone in collaboration.milestones)}

## Communication
- **Meeting Frequency**: {collaboration.meeting_frequency or 'TBD'}
- **Channels**: {', '.join(collaboration.communication_channels)}

## Resources Shared
{"".join(f"- {resource}\\n" for resource in collaboration.resources_shared)}

## Outcomes & Publications
{"".join(f"- {pub}\\n" for pub in collaboration.publications)}

## Project Notes
{collaboration.notes or 'Add project notes and updates here...'}

---
*Created*: {collaboration.created_at.strftime('%Y-%m-%d %H:%M')}  
*Last Updated*: {(collaboration.updated_at or collaboration.created_at).strftime('%Y-%m-%d %H:%M')}  
*Project ID*: {collaboration.id}
"""
        
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_collaborator(self, collaborator: Collaborator):
        """Save collaborator to JSON storage."""
        file_path = self.profiles_dir / f"{collaborator.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(collaborator.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_collaboration(self, collaboration: Collaboration):
        """Save collaboration to JSON storage."""
        file_path = self.projects_dir / f"{collaboration.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(collaboration.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_interaction(self, interaction: Interaction):
        """Save interaction to JSON storage."""
        file_path = self.interactions_dir / f"{interaction.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(interaction.to_dict(), f, indent=2, ensure_ascii=False)
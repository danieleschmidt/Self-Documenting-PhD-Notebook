"""
Collaborative Intelligence System for Research Networks

A next-generation system for enabling intelligent collaboration between researchers,
institutions, and AI agents across global research networks with real-time
knowledge sharing and distributed research coordination.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import hashlib
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CollaborationType(Enum):
    """Types of research collaboration."""
    PEER_REVIEW = "peer_review"
    CO_AUTHORING = "co_authoring"
    DATA_SHARING = "data_sharing"
    METHODOLOGY_EXCHANGE = "methodology_exchange"
    CROSS_INSTITUTIONAL = "cross_institutional"
    INTERDISCIPLINARY = "interdisciplinary"
    MENTORSHIP = "mentorship"
    OPEN_COLLABORATION = "open_collaboration"


class ExpertiseLevel(Enum):
    """Levels of expertise for collaboration matching."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    WORLD_CLASS = "world_class"


@dataclass
class ResearcherProfile:
    """Profile of a researcher for collaboration matching."""
    researcher_id: str
    name: str
    institution: str
    email: str
    primary_domain: str
    expertise_areas: List[str] = field(default_factory=list)
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    collaboration_history: List[str] = field(default_factory=list)
    availability_score: float = 1.0
    reputation_score: float = 0.5
    preferred_collaboration_types: List[CollaborationType] = field(default_factory=list)
    timezone: str = "UTC"
    languages: List[str] = field(default_factory=lambda: ["en"])
    publications: List[str] = field(default_factory=list)
    h_index: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CollaborationRequest:
    """Request for research collaboration."""
    request_id: str
    requester_id: str
    collaboration_type: CollaborationType
    topic: str
    description: str
    required_expertise: List[str]
    preferred_expertise_level: ExpertiseLevel
    timeline: timedelta
    max_collaborators: int = 5
    is_open: bool = True
    urgency_level: float = 0.5
    compensation_offered: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    responses: List[str] = field(default_factory=list)


@dataclass
class CollaborationMatch:
    """A potential collaboration match between researchers."""
    match_id: str
    request_id: str
    researcher_id: str
    compatibility_score: float
    expertise_match_score: float
    availability_match_score: float
    reputation_factor: float
    geographic_factor: float
    language_compatibility: float
    rationale: str
    confidence_level: float
    estimated_success_probability: float


@dataclass
class KnowledgeContribution:
    """A knowledge contribution to the collaborative network."""
    contribution_id: str
    contributor_id: str
    content_type: str  # "paper", "dataset", "code", "methodology", "insight"
    title: str
    description: str
    domain: str
    tags: List[str] = field(default_factory=list)
    access_level: str = "open"  # "open", "restricted", "private"
    quality_score: float = 0.0
    impact_score: float = 0.0
    citations: int = 0
    downloads: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborativeIntelligenceSystem:
    """
    Main system for managing intelligent research collaboration.
    
    Features:
    - Smart researcher matching based on expertise and compatibility
    - Real-time knowledge sharing and contribution tracking
    - Collaboration quality assessment and optimization
    - Cross-institutional research coordination
    - Automated peer review facilitation
    """
    
    def __init__(self, 
                 system_id: str = None,
                 enable_global_network: bool = True,
                 enable_ai_matching: bool = True):
        self.system_id = system_id or f"cis_{uuid.uuid4().hex[:8]}"
        self.enable_global_network = enable_global_network
        self.enable_ai_matching = enable_ai_matching
        
        # Core data structures
        self.researchers: Dict[str, ResearcherProfile] = {}
        self.collaboration_requests: Dict[str, CollaborationRequest] = {}
        self.active_collaborations: Dict[str, Dict] = {}
        self.knowledge_base: Dict[str, KnowledgeContribution] = {}
        self.collaboration_history: List[Dict] = []
        
        # Matching algorithms
        self.matching_algorithms = {
            "expertise_based": self._expertise_based_matching,
            "reputation_weighted": self._reputation_weighted_matching,
            "availability_optimized": self._availability_optimized_matching,
            "ai_enhanced": self._ai_enhanced_matching
        }
        
        # Performance metrics
        self.metrics = {
            "total_matches_made": 0,
            "successful_collaborations": 0,
            "knowledge_contributions": 0,
            "average_collaboration_duration": 0,
            "cross_institutional_rate": 0.0,
            "interdisciplinary_rate": 0.0
        }
        
        logger.info(f"Initialized Collaborative Intelligence System: {self.system_id}")
    
    async def register_researcher(self, profile: ResearcherProfile) -> bool:
        """Register a new researcher in the system."""
        try:
            # Validate profile
            if not profile.researcher_id or not profile.name:
                raise ValueError("Researcher ID and name are required")
            
            # Check for duplicates
            if profile.researcher_id in self.researchers:
                logger.warning(f"Researcher {profile.researcher_id} already registered")
                return False
            
            # Store profile
            self.researchers[profile.researcher_id] = profile
            
            # Trigger matching for existing requests
            if self.enable_ai_matching:
                await self._trigger_matching_for_new_researcher(profile)
            
            logger.info(f"Registered researcher: {profile.name} ({profile.researcher_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register researcher: {e}")
            return False
    
    async def submit_collaboration_request(self, request: CollaborationRequest) -> str:
        """Submit a new collaboration request."""
        try:
            # Generate ID if not provided
            if not request.request_id:
                request.request_id = f"collab_{uuid.uuid4().hex[:12]}"
            
            # Validate request
            if not request.requester_id or request.requester_id not in self.researchers:
                raise ValueError("Invalid requester ID")
            
            # Store request
            self.collaboration_requests[request.request_id] = request
            
            # Find potential matches
            matches = await self.find_collaboration_matches(request.request_id)
            
            # Notify potential collaborators
            if matches and self.enable_global_network:
                await self._notify_potential_collaborators(request, matches[:10])
            
            logger.info(f"Submitted collaboration request: {request.request_id}")
            return request.request_id
            
        except Exception as e:
            logger.error(f"Failed to submit collaboration request: {e}")
            raise
    
    async def find_collaboration_matches(self, request_id: str) -> List[CollaborationMatch]:
        """Find potential collaboration matches for a request."""
        try:
            request = self.collaboration_requests.get(request_id)
            if not request:
                raise ValueError(f"Request {request_id} not found")
            
            matches = []
            
            # Use different matching algorithms
            for algorithm_name, algorithm in self.matching_algorithms.items():
                if algorithm_name == "ai_enhanced" and not self.enable_ai_matching:
                    continue
                
                algorithm_matches = await algorithm(request)
                matches.extend(algorithm_matches)
            
            # Remove duplicates and sort by score
            unique_matches = {}
            for match in matches:
                if match.researcher_id not in unique_matches:
                    unique_matches[match.researcher_id] = match
                else:
                    # Keep the match with higher score
                    if match.compatibility_score > unique_matches[match.researcher_id].compatibility_score:
                        unique_matches[match.researcher_id] = match
            
            sorted_matches = sorted(unique_matches.values(), 
                                  key=lambda x: x.compatibility_score, reverse=True)
            
            logger.info(f"Found {len(sorted_matches)} matches for request {request_id}")
            return sorted_matches
            
        except Exception as e:
            logger.error(f"Failed to find matches: {e}")
            return []
    
    async def _expertise_based_matching(self, request: CollaborationRequest) -> List[CollaborationMatch]:
        """Match based on expertise overlap."""
        matches = []
        
        for researcher_id, researcher in self.researchers.items():
            if researcher_id == request.requester_id:
                continue
            
            # Calculate expertise match
            expertise_overlap = len(set(request.required_expertise) & 
                                  set(researcher.expertise_areas))
            total_required = len(request.required_expertise)
            expertise_score = expertise_overlap / max(total_required, 1)
            
            # Check expertise level compatibility
            level_scores = {
                ExpertiseLevel.NOVICE: 1,
                ExpertiseLevel.INTERMEDIATE: 2,
                ExpertiseLevel.ADVANCED: 3,
                ExpertiseLevel.EXPERT: 4,
                ExpertiseLevel.WORLD_CLASS: 5
            }
            
            required_level = level_scores[request.preferred_expertise_level]
            researcher_level = level_scores[researcher.expertise_level]
            level_compatibility = min(1.0, researcher_level / required_level)
            
            if expertise_score > 0.3:  # Minimum threshold
                compatibility_score = (expertise_score * 0.7 + level_compatibility * 0.3)
                
                match = CollaborationMatch(
                    match_id=f"match_{uuid.uuid4().hex[:8]}",
                    request_id=request.request_id,
                    researcher_id=researcher_id,
                    compatibility_score=compatibility_score,
                    expertise_match_score=expertise_score,
                    availability_match_score=researcher.availability_score,
                    reputation_factor=researcher.reputation_score,
                    geographic_factor=1.0,  # Would calculate based on location
                    language_compatibility=1.0,  # Would check language overlap
                    rationale=f"Expertise overlap: {expertise_overlap}/{total_required} areas",
                    confidence_level=expertise_score,
                    estimated_success_probability=compatibility_score * 0.8
                )
                matches.append(match)
        
        return matches
    
    async def _reputation_weighted_matching(self, request: CollaborationRequest) -> List[CollaborationMatch]:
        """Match based on reputation and track record."""
        matches = []
        
        for researcher_id, researcher in self.researchers.items():
            if researcher_id == request.requester_id:
                continue
            
            # Base expertise matching
            expertise_overlap = len(set(request.required_expertise) & 
                                  set(researcher.expertise_areas))
            if expertise_overlap == 0:
                continue
            
            # Reputation factors
            reputation_score = researcher.reputation_score
            h_index_factor = min(1.0, researcher.h_index / 50.0)  # Normalize h-index
            collaboration_experience = len(researcher.collaboration_history) / 20.0
            
            # Combined score
            compatibility_score = (
                (expertise_overlap / len(request.required_expertise)) * 0.4 +
                reputation_score * 0.3 +
                h_index_factor * 0.2 +
                min(1.0, collaboration_experience) * 0.1
            )
            
            if compatibility_score > 0.4:
                match = CollaborationMatch(
                    match_id=f"match_{uuid.uuid4().hex[:8]}",
                    request_id=request.request_id,
                    researcher_id=researcher_id,
                    compatibility_score=compatibility_score,
                    expertise_match_score=expertise_overlap / len(request.required_expertise),
                    availability_match_score=researcher.availability_score,
                    reputation_factor=reputation_score,
                    geographic_factor=1.0,
                    language_compatibility=1.0,
                    rationale=f"High reputation researcher (H-index: {researcher.h_index})",
                    confidence_level=reputation_score,
                    estimated_success_probability=compatibility_score * 0.9
                )
                matches.append(match)
        
        return matches
    
    async def _availability_optimized_matching(self, request: CollaborationRequest) -> List[CollaborationMatch]:
        """Match based on availability and timeline compatibility."""
        matches = []
        
        for researcher_id, researcher in self.researchers.items():
            if researcher_id == request.requester_id:
                continue
            
            # Check basic expertise compatibility
            expertise_overlap = len(set(request.required_expertise) & 
                                  set(researcher.expertise_areas))
            if expertise_overlap == 0:
                continue
            
            # Availability scoring
            availability_score = researcher.availability_score
            
            # Timeline compatibility (simplified)
            timeline_compatibility = 1.0  # Would implement actual calendar checking
            
            compatibility_score = (
                (expertise_overlap / len(request.required_expertise)) * 0.5 +
                availability_score * 0.3 +
                timeline_compatibility * 0.2
            )
            
            if availability_score > 0.6:  # Only highly available researchers
                match = CollaborationMatch(
                    match_id=f"match_{uuid.uuid4().hex[:8]}",
                    request_id=request.request_id,
                    researcher_id=researcher_id,
                    compatibility_score=compatibility_score,
                    expertise_match_score=expertise_overlap / len(request.required_expertise),
                    availability_match_score=availability_score,
                    reputation_factor=researcher.reputation_score,
                    geographic_factor=1.0,
                    language_compatibility=1.0,
                    rationale=f"High availability (score: {availability_score:.2f})",
                    confidence_level=availability_score,
                    estimated_success_probability=compatibility_score * availability_score
                )
                matches.append(match)
        
        return matches
    
    async def _ai_enhanced_matching(self, request: CollaborationRequest) -> List[CollaborationMatch]:
        """Advanced AI-based matching considering multiple factors."""
        matches = []
        
        # This would integrate with advanced AI models
        # For now, implementing a sophisticated scoring system
        
        for researcher_id, researcher in self.researchers.items():
            if researcher_id == request.requester_id:
                continue
            
            # Multi-factor compatibility assessment
            expertise_score = self._calculate_expertise_compatibility(request, researcher)
            if expertise_score < 0.2:
                continue
            
            # Collaboration history analysis
            history_score = self._analyze_collaboration_history(researcher, request)
            
            # Domain transfer potential
            transfer_score = self._calculate_domain_transfer_potential(request, researcher)
            
            # Personality and working style compatibility (placeholder)
            personality_score = 0.7  # Would use personality matching algorithms
            
            # Time zone and logistics compatibility
            logistics_score = self._calculate_logistics_compatibility(researcher)
            
            # AI-weighted combination
            weights = {
                'expertise': 0.35,
                'history': 0.2,
                'transfer': 0.15,
                'personality': 0.15,
                'logistics': 0.15
            }
            
            compatibility_score = (
                expertise_score * weights['expertise'] +
                history_score * weights['history'] +
                transfer_score * weights['transfer'] +
                personality_score * weights['personality'] +
                logistics_score * weights['logistics']
            )
            
            if compatibility_score > 0.5:
                match = CollaborationMatch(
                    match_id=f"match_{uuid.uuid4().hex[:8]}",
                    request_id=request.request_id,
                    researcher_id=researcher_id,
                    compatibility_score=compatibility_score,
                    expertise_match_score=expertise_score,
                    availability_match_score=researcher.availability_score,
                    reputation_factor=researcher.reputation_score,
                    geographic_factor=logistics_score,
                    language_compatibility=1.0,
                    rationale=f"AI-enhanced matching (confidence: {compatibility_score:.2f})",
                    confidence_level=compatibility_score,
                    estimated_success_probability=compatibility_score
                )
                matches.append(match)
        
        return matches
    
    def _calculate_expertise_compatibility(self, request: CollaborationRequest, 
                                         researcher: ResearcherProfile) -> float:
        """Calculate expertise compatibility score."""
        required_expertise = set(request.required_expertise)
        researcher_expertise = set(researcher.expertise_areas)
        
        # Direct overlap
        overlap = len(required_expertise & researcher_expertise)
        direct_score = overlap / max(len(required_expertise), 1)
        
        # Complementary expertise bonus
        complementary_score = len(researcher_expertise - required_expertise) / 10.0
        
        return min(1.0, direct_score + complementary_score * 0.2)
    
    def _analyze_collaboration_history(self, researcher: ResearcherProfile, 
                                     request: CollaborationRequest) -> float:
        """Analyze researcher's collaboration history for compatibility."""
        if not researcher.collaboration_history:
            return 0.5  # Neutral score for no history
        
        # Recent collaboration frequency
        recent_collaborations = len(researcher.collaboration_history)
        frequency_score = min(1.0, recent_collaborations / 10.0)
        
        # Success rate (placeholder - would track actual outcomes)
        success_rate = 0.8  # Assumed success rate
        
        return frequency_score * success_rate
    
    def _calculate_domain_transfer_potential(self, request: CollaborationRequest, 
                                           researcher: ResearcherProfile) -> float:
        """Calculate potential for cross-domain knowledge transfer."""
        # This would implement sophisticated domain similarity analysis
        # For now, return a baseline score
        return 0.6
    
    def _calculate_logistics_compatibility(self, researcher: ResearcherProfile) -> float:
        """Calculate logistics compatibility (timezone, location, etc.)."""
        # Simplified implementation
        return researcher.availability_score
    
    async def contribute_knowledge(self, contribution: KnowledgeContribution) -> str:
        """Add a knowledge contribution to the collaborative network."""
        try:
            if not contribution.contribution_id:
                contribution.contribution_id = f"contrib_{uuid.uuid4().hex[:12]}"
            
            self.knowledge_base[contribution.contribution_id] = contribution
            self.metrics["knowledge_contributions"] += 1
            
            # Index contribution for search
            await self._index_knowledge_contribution(contribution)
            
            logger.info(f"Added knowledge contribution: {contribution.title}")
            return contribution.contribution_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge contribution: {e}")
            raise
    
    async def search_knowledge_base(self, query: str, domain: str = None, 
                                  content_type: str = None) -> List[KnowledgeContribution]:
        """Search the collaborative knowledge base."""
        try:
            results = []
            
            for contrib_id, contribution in self.knowledge_base.items():
                # Basic text matching (would use advanced search in production)
                if query.lower() in contribution.title.lower() or \
                   query.lower() in contribution.description.lower() or \
                   any(query.lower() in tag.lower() for tag in contribution.tags):
                    
                    # Apply filters
                    if domain and contribution.domain != domain:
                        continue
                    if content_type and contribution.content_type != content_type:
                        continue
                    
                    results.append(contribution)
            
            # Sort by relevance (simplified)
            results.sort(key=lambda x: x.quality_score + x.impact_score, reverse=True)
            
            return results[:20]  # Return top 20 results
            
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return []
    
    async def _trigger_matching_for_new_researcher(self, profile: ResearcherProfile):
        """Trigger matching process for a newly registered researcher."""
        try:
            for request_id in self.collaboration_requests:
                matches = await self.find_collaboration_matches(request_id)
                # Filter for this researcher
                new_matches = [m for m in matches if m.researcher_id == profile.researcher_id]
                
                if new_matches and self.enable_global_network:
                    await self._notify_of_new_matches(request_id, new_matches)
                    
        except Exception as e:
            logger.error(f"Failed to trigger matching for new researcher: {e}")
    
    async def _notify_potential_collaborators(self, request: CollaborationRequest, 
                                           matches: List[CollaborationMatch]):
        """Notify potential collaborators about a request."""
        # Implementation would send actual notifications
        logger.info(f"Notifying {len(matches)} potential collaborators for request {request.request_id}")
    
    async def _notify_of_new_matches(self, request_id: str, matches: List[CollaborationMatch]):
        """Notify about new potential matches."""
        logger.info(f"New matches found for request {request_id}: {len(matches)}")
    
    async def _index_knowledge_contribution(self, contribution: KnowledgeContribution):
        """Index a knowledge contribution for efficient search."""
        # Would implement advanced indexing (vector embeddings, etc.)
        pass
    
    def get_collaboration_analytics(self) -> Dict[str, Any]:
        """Get analytics about collaboration patterns."""
        return {
            "system_metrics": self.metrics,
            "active_researchers": len(self.researchers),
            "active_requests": len([r for r in self.collaboration_requests.values() if r.is_open]),
            "knowledge_contributions": len(self.knowledge_base),
            "domains_represented": len(set(r.primary_domain for r in self.researchers.values())),
            "collaboration_success_rate": (
                self.metrics["successful_collaborations"] / 
                max(self.metrics["total_matches_made"], 1)
            ),
            "average_researcher_expertise_areas": (
                sum(len(r.expertise_areas) for r in self.researchers.values()) / 
                max(len(self.researchers), 1)
            )
        }
    
    def export_collaboration_network(self, file_path: str):
        """Export the collaboration network for analysis."""
        try:
            network_data = {
                "researchers": {r_id: {
                    "name": r.name,
                    "institution": r.institution,
                    "domain": r.primary_domain,
                    "expertise_areas": r.expertise_areas,
                    "collaboration_count": len(r.collaboration_history)
                } for r_id, r in self.researchers.items()},
                "requests": {r_id: {
                    "type": r.collaboration_type.value,
                    "topic": r.topic,
                    "requester": r.requester_id,
                    "responses": len(r.responses)
                } for r_id, r in self.collaboration_requests.items()},
                "knowledge_base": {k_id: {
                    "title": k.title,
                    "type": k.content_type,
                    "domain": k.domain,
                    "quality": k.quality_score
                } for k_id, k in self.knowledge_base.items()},
                "analytics": self.get_collaboration_analytics(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(network_data, f, indent=2)
            
            logger.info(f"Exported collaboration network to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export collaboration network: {e}")
            raise


# Convenience functions for integration with the main notebook system

async def setup_collaborative_research(notebook) -> CollaborativeIntelligenceSystem:
    """Set up collaborative research capabilities for a notebook."""
    try:
        # Initialize collaboration system
        collab_system = CollaborativeIntelligenceSystem(
            system_id=f"notebook_{notebook.vault_path.name}",
            enable_global_network=True,
            enable_ai_matching=True
        )
        
        # Create researcher profile from notebook info
        profile = ResearcherProfile(
            researcher_id=f"user_{notebook.author.replace(' ', '_').lower()}",
            name=notebook.author,
            institution=notebook.institution,
            email="",  # Would get from config
            primary_domain=notebook.field,
            expertise_areas=[notebook.field, notebook.subfield] if notebook.subfield else [notebook.field],
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            timezone="UTC"  # Would detect from system
        )
        
        await collab_system.register_researcher(profile)
        
        return collab_system
        
    except Exception as e:
        logger.error(f"Failed to setup collaborative research: {e}")
        raise


def create_collaboration_request(topic: str, description: str, 
                               required_expertise: List[str],
                               collaboration_type: CollaborationType = CollaborationType.CO_AUTHORING,
                               requester_id: str = None) -> CollaborationRequest:
    """Create a collaboration request."""
    return CollaborationRequest(
        request_id="",  # Will be generated
        requester_id=requester_id,
        collaboration_type=collaboration_type,
        topic=topic,
        description=description,
        required_expertise=required_expertise,
        preferred_expertise_level=ExpertiseLevel.INTERMEDIATE,
        timeline=timedelta(days=90),
        max_collaborators=3
    )
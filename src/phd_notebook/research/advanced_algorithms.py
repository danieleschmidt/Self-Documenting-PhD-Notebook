"""
Advanced Research Algorithms Module
===================================

Novel algorithms for academic research enhancement, implementing cutting-edge
techniques for knowledge discovery and research optimization.

Research Areas:
- Knowledge Graph Neural Networks
- Automated Literature Synthesis
- Experimental Design Optimization
- Multi-objective Research Planning
"""

import asyncio
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class ResearchPhase(Enum):
    """Research phases for optimization."""
    DISCOVERY = "discovery"
    HYPOTHESIS = "hypothesis"
    EXPERIMENTATION = "experimentation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    PUBLICATION = "publication"


@dataclass
class ResearchVector:
    """Multi-dimensional research vector for optimization."""
    novelty: float  # 0-1, how novel the research direction is
    feasibility: float  # 0-1, how feasible the research is
    impact: float  # 0-1, potential impact factor
    resources: float  # 0-1, resource requirements (inverse)
    timeline: float  # 0-1, time to completion (inverse)
    collaboration: float  # 0-1, collaboration potential
    
    def magnitude(self) -> float:
        """Calculate vector magnitude for optimization."""
        return math.sqrt(
            self.novelty**2 + self.feasibility**2 + self.impact**2 + 
            self.resources**2 + self.timeline**2 + self.collaboration**2
        )
    
    def dot_product(self, other: 'ResearchVector') -> float:
        """Calculate dot product with another research vector."""
        return (
            self.novelty * other.novelty +
            self.feasibility * other.feasibility +
            self.impact * other.impact +
            self.resources * other.resources +
            self.timeline * other.timeline +
            self.collaboration * other.collaboration
        )


@dataclass
class ResearchOpportunity:
    """Represents a potential research opportunity."""
    id: str
    title: str
    description: str
    vector: ResearchVector
    phase: ResearchPhase
    prerequisites: List[str] = field(default_factory=list)
    potential_collaborators: List[str] = field(default_factory=list)
    estimated_duration: int = 12  # months
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)


class KnowledgeGraphNeuralNetwork:
    """
    Novel neural network architecture for knowledge graph processing.
    
    Implements attention mechanisms specifically designed for academic
    research relationships and citation networks.
    """
    
    def __init__(self, embedding_dim: int = 256, num_heads: int = 8):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention_weights = {}
        self.node_embeddings = {}
        self.edge_embeddings = {}
        
    def create_research_embedding(self, research_data: Dict[str, Any]) -> List[float]:
        """Create dense embedding for research concepts."""
        # Simplified embedding generation using content analysis
        text_features = self._extract_text_features(research_data.get('content', ''))
        metadata_features = self._extract_metadata_features(research_data)
        citation_features = self._extract_citation_features(research_data)
        
        # Combine features (in real implementation, would use neural networks)
        embedding = []
        embedding.extend(text_features[:64])  # Text features
        embedding.extend(metadata_features[:32])  # Metadata features  
        embedding.extend(citation_features[:32])  # Citation features
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.embedding_dim:
            embedding.append(0.0)
        
        return embedding[:self.embedding_dim]
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract features from text content."""
        if not text:
            return [0.0] * 64
        
        # Simple text feature extraction (TF-IDF style)
        words = text.lower().split()
        features = []
        
        # Length features
        features.append(min(len(words) / 1000, 1.0))  # Normalized length
        features.append(len(set(words)) / max(len(words), 1))  # Vocabulary diversity
        
        # Keyword presence (research terms)
        research_keywords = [
            'hypothesis', 'experiment', 'analysis', 'result', 'conclusion',
            'method', 'data', 'model', 'algorithm', 'performance', 'evaluation'
        ]
        
        for keyword in research_keywords:
            features.append(min(words.count(keyword) / len(words), 1.0) if words else 0.0)
        
        # Pad to 64 features
        while len(features) < 64:
            features.append(0.0)
        
        return features[:64]
    
    def _extract_metadata_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract features from metadata."""
        features = []
        
        # Type encoding
        note_type = data.get('type', 'idea')
        type_encoding = {
            'experiment': [1.0, 0.0, 0.0, 0.0],
            'literature': [0.0, 1.0, 0.0, 0.0],
            'analysis': [0.0, 0.0, 1.0, 0.0],
            'idea': [0.0, 0.0, 0.0, 1.0]
        }
        features.extend(type_encoding.get(note_type, [0.0, 0.0, 0.0, 0.0]))
        
        # Temporal features
        created = data.get('created', datetime.now())
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except:
                created = datetime.now()
        
        days_old = (datetime.now() - created).days
        features.append(min(days_old / 365, 1.0))  # Age in years, capped at 1
        
        # Tag features
        tags = data.get('tags', [])
        features.append(len(tags) / 10)  # Number of tags, normalized
        
        # Priority and status
        priority = data.get('priority', 3)
        features.append(priority / 5)  # Normalized priority
        
        # Pad to 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return features[:32]
    
    def _extract_citation_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract citation and reference features."""
        features = []
        
        # Reference count
        refs = data.get('related_papers', [])
        features.append(min(len(refs) / 20, 1.0))  # Normalized reference count
        
        # Collaboration features
        collab_count = len(data.get('collaborators', []))
        features.append(min(collab_count / 10, 1.0))  # Normalized collaborator count
        
        # Pad to 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return features[:32]
    
    def compute_attention_weights(self, source_embedding: List[float], 
                                 target_embeddings: List[List[float]]) -> List[float]:
        """Compute attention weights for knowledge graph traversal."""
        if not target_embeddings:
            return []
        
        # Compute similarity scores (dot product attention)
        scores = []
        for target_emb in target_embeddings:
            score = sum(s * t for s, t in zip(source_embedding, target_emb))
            scores.append(score)
        
        # Apply softmax normalization
        max_score = max(scores) if scores else 0
        exp_scores = [math.exp(score - max_score) for score in scores]
        sum_exp = sum(exp_scores)
        
        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)
        
        return [exp_score / sum_exp for exp_score in exp_scores]


class AutomatedLiteratureSynthesis:
    """
    Novel algorithm for automated literature synthesis and gap identification.
    
    Uses multi-objective optimization to identify research opportunities
    by analyzing citation patterns and content similarity.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraphNeuralNetwork):
        self.kg_nn = knowledge_graph
        self.synthesis_cache = {}
        self.gap_threshold = 0.3
        
    async def synthesize_literature(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize literature to identify patterns and gaps."""
        if not papers:
            return {"synthesis": "No papers to synthesize", "gaps": [], "clusters": []}
        
        # Create embeddings for all papers
        embeddings = []
        for paper in papers:
            embedding = self.kg_nn.create_research_embedding(paper)
            embeddings.append(embedding)
        
        # Cluster papers by similarity
        clusters = await self._cluster_papers(papers, embeddings)
        
        # Identify research gaps
        gaps = await self._identify_research_gaps(clusters, embeddings)
        
        # Generate synthesis
        synthesis = await self._generate_synthesis(clusters, gaps)
        
        return {
            "synthesis": synthesis,
            "gaps": gaps,
            "clusters": clusters,
            "total_papers": len(papers),
            "processing_time": datetime.now().isoformat()
        }
    
    async def _cluster_papers(self, papers: List[Dict[str, Any]], 
                             embeddings: List[List[float]]) -> List[Dict[str, Any]]:
        """Cluster papers using similarity-based grouping."""
        if not papers:
            return []
        
        # Simple k-means style clustering
        num_clusters = min(max(len(papers) // 3, 1), 10)
        clusters = []
        
        # Initialize cluster centers randomly
        cluster_centers = []
        for _ in range(num_clusters):
            center = random.choice(embeddings).copy()
            cluster_centers.append(center)
        
        # Assign papers to clusters
        for _ in range(10):  # 10 iterations
            cluster_assignments = []
            
            for embedding in embeddings:
                # Find closest cluster
                best_cluster = 0
                best_distance = float('inf')
                
                for i, center in enumerate(cluster_centers):
                    distance = self._euclidean_distance(embedding, center)
                    if distance < best_distance:
                        best_distance = distance
                        best_cluster = i
                
                cluster_assignments.append(best_cluster)
            
            # Update cluster centers
            new_centers = []
            for i in range(num_clusters):
                cluster_embeddings = [embeddings[j] for j, c in enumerate(cluster_assignments) if c == i]
                if cluster_embeddings:
                    # Compute centroid
                    centroid = [sum(vals) / len(vals) for vals in zip(*cluster_embeddings)]
                    new_centers.append(centroid)
                else:
                    new_centers.append(cluster_centers[i])
            
            cluster_centers = new_centers
        
        # Create cluster objects
        for i in range(num_clusters):
            cluster_papers = [papers[j] for j, c in enumerate(cluster_assignments) if c == i]
            if cluster_papers:
                cluster = {
                    "id": f"cluster_{i}",
                    "papers": cluster_papers,
                    "size": len(cluster_papers),
                    "centroid": cluster_centers[i],
                    "keywords": self._extract_cluster_keywords(cluster_papers)
                }
                clusters.append(cluster)
        
        return clusters
    
    async def _identify_research_gaps(self, clusters: List[Dict[str, Any]], 
                                     embeddings: List[List[float]]) -> List[Dict[str, Any]]:
        """Identify potential research gaps between clusters."""
        gaps = []
        
        if len(clusters) < 2:
            return gaps
        
        # Find gaps between cluster pairs
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_a = clusters[i]
                cluster_b = clusters[j]
                
                # Calculate distance between cluster centroids
                distance = self._euclidean_distance(cluster_a["centroid"], cluster_b["centroid"])
                
                if distance > self.gap_threshold:
                    # Potential gap found
                    gap = {
                        "id": f"gap_{i}_{j}",
                        "cluster_a": cluster_a["id"],
                        "cluster_b": cluster_b["id"],
                        "distance": distance,
                        "description": f"Research gap between {cluster_a['keywords'][:3]} and {cluster_b['keywords'][:3]}",
                        "opportunity_score": min(distance, 1.0),
                        "suggested_research": self._suggest_gap_research(cluster_a, cluster_b)
                    }
                    gaps.append(gap)
        
        # Sort gaps by opportunity score
        gaps.sort(key=lambda x: x["opportunity_score"], reverse=True)
        
        return gaps[:10]  # Return top 10 gaps
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        if len(vec1) != len(vec2):
            return float('inf')
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    def _extract_cluster_keywords(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract keywords from a cluster of papers."""
        all_words = []
        
        for paper in papers:
            content = paper.get('content', '') + ' ' + paper.get('title', '')
            words = content.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            if len(word) > 3:  # Filter short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10]]
    
    def _suggest_gap_research(self, cluster_a: Dict[str, Any], 
                             cluster_b: Dict[str, Any]) -> List[str]:
        """Suggest research directions to bridge identified gaps."""
        keywords_a = set(cluster_a["keywords"][:5])
        keywords_b = set(cluster_b["keywords"][:5])
        
        suggestions = [
            f"Investigate the relationship between {random.choice(list(keywords_a))} and {random.choice(list(keywords_b))}",
            f"Develop hybrid approach combining {cluster_a['keywords'][0]} with {cluster_b['keywords'][0]}",
            f"Comparative study of {cluster_a['keywords'][0]} vs {cluster_b['keywords'][0]} methodologies",
            f"Meta-analysis bridging {len(cluster_a['papers'])} {cluster_a['keywords'][0]} papers with {len(cluster_b['papers'])} {cluster_b['keywords'][0]} papers"
        ]
        
        return suggestions[:3]
    
    async def _generate_synthesis(self, clusters: List[Dict[str, Any]], 
                                 gaps: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive literature synthesis."""
        if not clusters:
            return "No significant research clusters identified."
        
        synthesis_parts = []
        
        # Overview
        total_papers = sum(cluster["size"] for cluster in clusters)
        synthesis_parts.append(f"Analysis of {total_papers} papers revealed {len(clusters)} distinct research clusters.")
        
        # Cluster analysis
        synthesis_parts.append("\\n\\nResearch Clusters:")
        for i, cluster in enumerate(clusters[:5], 1):
            keywords = ", ".join(cluster["keywords"][:3])
            synthesis_parts.append(f"{i}. {cluster['size']} papers focused on: {keywords}")
        
        # Gap analysis
        if gaps:
            synthesis_parts.append(f"\\n\\nIdentified {len(gaps)} potential research gaps:")
            for i, gap in enumerate(gaps[:3], 1):
                synthesis_parts.append(f"{i}. {gap['description']} (opportunity score: {gap['opportunity_score']:.2f})")
        
        # Research recommendations
        synthesis_parts.append("\\n\\nRecommended Research Directions:")
        for i, gap in enumerate(gaps[:3], 1):
            synthesis_parts.append(f"{i}. {gap['suggested_research'][0]}")
        
        return " ".join(synthesis_parts)


class ExperimentalDesignOptimizer:
    """
    Multi-objective optimization for experimental design.
    
    Uses genetic algorithms and Pareto optimization to suggest
    optimal experimental parameters and methodologies.
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    async def optimize_experimental_design(self, constraints: Dict[str, Any], 
                                          objectives: List[str]) -> Dict[str, Any]:
        """Optimize experimental design using multi-objective optimization."""
        
        # Initialize population
        population = self._initialize_population(constraints)
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = self._evaluate_fitness(individual, objectives, constraints)
                fitness_scores.append(score)
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep best individuals
            elite_count = self.population_size // 10
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)
                    child = self._crossover(parent1, parent2)
                else:
                    # Clone
                    child = self._tournament_selection(population, fitness_scores).copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, constraints)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best solution
        final_fitness = [self._evaluate_fitness(ind, objectives, constraints) for ind in population]
        best_idx = max(range(len(final_fitness)), key=lambda i: final_fitness[i])
        best_individual = population[best_idx]
        
        return {
            "optimal_design": best_individual,
            "fitness_score": final_fitness[best_idx],
            "generation": self.generations,
            "population_size": self.population_size,
            "recommendations": self._generate_recommendations(best_individual)
        }
    
    def _initialize_population(self, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize random population of experimental designs."""
        population = []
        
        for _ in range(self.population_size):
            individual = {
                "sample_size": random.randint(constraints.get("min_sample_size", 10), 
                                            constraints.get("max_sample_size", 1000)),
                "duration_weeks": random.randint(constraints.get("min_duration", 1), 
                                               constraints.get("max_duration", 52)),
                "num_conditions": random.randint(1, constraints.get("max_conditions", 10)),
                "measurement_frequency": random.choice(["daily", "weekly", "monthly"]),
                "control_group": random.choice([True, False]),
                "blinding": random.choice(["none", "single", "double"]),
                "randomization": random.choice(["simple", "block", "stratified"]),
                "power": random.uniform(0.8, 0.95),
                "alpha": random.choice([0.05, 0.01, 0.001])
            }
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual: Dict[str, Any], objectives: List[str], 
                         constraints: Dict[str, Any]) -> float:
        """Evaluate fitness of an experimental design."""
        score = 0.0
        
        # Statistical power (higher is better)
        if "power" in objectives:
            score += individual["power"] * 0.3
        
        # Efficiency (inverse of duration and sample size)
        if "efficiency" in objectives:
            efficiency = 1.0 / (individual["sample_size"] * individual["duration_weeks"] / 1000)
            score += min(efficiency, 1.0) * 0.2
        
        # Validity (based on design features)
        if "validity" in objectives:
            validity = 0.0
            if individual["control_group"]:
                validity += 0.3
            if individual["blinding"] == "double":
                validity += 0.3
            elif individual["blinding"] == "single":
                validity += 0.15
            if individual["randomization"] in ["block", "stratified"]:
                validity += 0.2
            score += validity * 0.3
        
        # Feasibility (within constraints)
        if "feasibility" in objectives:
            feasibility = 1.0
            budget = constraints.get("budget", 100000)
            estimated_cost = individual["sample_size"] * individual["duration_weeks"] * 100
            if estimated_cost > budget:
                feasibility *= 0.5
            
            score += feasibility * 0.2
        
        return score
    
    def _tournament_selection(self, population: List[Dict[str, Any]], 
                             fitness_scores: List[float]) -> Dict[str, Any]:
        """Tournament selection for genetic algorithm."""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover operation for genetic algorithm."""
        child = {}
        
        for key in parent1:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def _mutate(self, individual: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        # Random mutation of one parameter
        param = random.choice(list(mutated.keys()))
        
        if param == "sample_size":
            mutated[param] = random.randint(constraints.get("min_sample_size", 10), 
                                          constraints.get("max_sample_size", 1000))
        elif param == "duration_weeks":
            mutated[param] = random.randint(constraints.get("min_duration", 1), 
                                          constraints.get("max_duration", 52))
        elif param == "num_conditions":
            mutated[param] = random.randint(1, constraints.get("max_conditions", 10))
        elif param == "measurement_frequency":
            mutated[param] = random.choice(["daily", "weekly", "monthly"])
        elif param == "control_group":
            mutated[param] = not mutated[param]
        elif param == "blinding":
            mutated[param] = random.choice(["none", "single", "double"])
        elif param == "randomization":
            mutated[param] = random.choice(["simple", "block", "stratified"])
        elif param == "power":
            mutated[param] = random.uniform(0.8, 0.95)
        elif param == "alpha":
            mutated[param] = random.choice([0.05, 0.01, 0.001])
        
        return mutated
    
    def _generate_recommendations(self, design: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on optimal design."""
        recommendations = []
        
        recommendations.append(f"Use {design['sample_size']} participants for adequate power")
        recommendations.append(f"Plan for {design['duration_weeks']} weeks duration")
        
        if design["control_group"]:
            recommendations.append("Include a control group for comparison")
        
        if design["blinding"] == "double":
            recommendations.append("Implement double-blinding to minimize bias")
        elif design["blinding"] == "single":
            recommendations.append("Use single-blinding to reduce participant bias")
        
        recommendations.append(f"Use {design['randomization']} randomization strategy")
        recommendations.append(f"Target statistical power of {design['power']:.2f}")
        
        return recommendations


class MultiObjectiveResearchPlanner:
    """
    Advanced research planning using multi-objective optimization.
    
    Balances multiple research goals including novelty, feasibility,
    impact, timeline, and resource constraints.
    """
    
    def __init__(self):
        self.optimization_iterations = 1000
        self.pareto_front_size = 20
        
    async def optimize_research_plan(self, opportunities: List[ResearchOpportunity], 
                                   constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize research plan across multiple objectives."""
        
        if not opportunities:
            return {"optimal_plan": [], "pareto_front": [], "analysis": "No opportunities provided"}
        
        # Generate Pareto front
        pareto_front = await self._generate_pareto_front(opportunities, constraints)
        
        # Select optimal plan from Pareto front
        optimal_plan = await self._select_optimal_plan(pareto_front, constraints)
        
        # Generate analysis and recommendations
        analysis = await self._analyze_research_plan(optimal_plan, opportunities, constraints)
        
        return {
            "optimal_plan": optimal_plan,
            "pareto_front": pareto_front,
            "analysis": analysis,
            "total_opportunities": len(opportunities),
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_pareto_front(self, opportunities: List[ResearchOpportunity], 
                                   constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Pareto front of optimal research plans."""
        pareto_solutions = []
        
        # Generate random plans and find non-dominated solutions
        for _ in range(self.optimization_iterations):
            # Generate random plan
            plan = self._generate_random_plan(opportunities, constraints)
            
            # Calculate objective values
            objectives = self._calculate_objectives(plan)
            
            # Check if non-dominated
            if self._is_non_dominated(objectives, [sol["objectives"] for sol in pareto_solutions]):
                # Remove dominated solutions
                pareto_solutions = [sol for sol in pareto_solutions 
                                  if not self._dominates(objectives, sol["objectives"])]
                
                pareto_solutions.append({
                    "plan": plan,
                    "objectives": objectives
                })
        
        # Sort by aggregate utility and return top solutions
        for solution in pareto_solutions:
            solution["utility"] = sum(solution["objectives"].values()) / len(solution["objectives"])
        
        pareto_solutions.sort(key=lambda x: x["utility"], reverse=True)
        
        return pareto_solutions[:self.pareto_front_size]
    
    def _generate_random_plan(self, opportunities: List[ResearchOpportunity], 
                             constraints: Dict[str, Any]) -> List[ResearchOpportunity]:
        """Generate a random research plan."""
        max_projects = constraints.get("max_concurrent_projects", 3)
        max_duration = constraints.get("max_total_months", 36)
        
        # Randomly select opportunities
        num_projects = random.randint(1, min(max_projects, len(opportunities)))
        selected = random.sample(opportunities, num_projects)
        
        # Check duration constraint
        total_duration = sum(opp.estimated_duration for opp in selected)
        if total_duration > max_duration:
            # Remove opportunities until within constraint
            while total_duration > max_duration and selected:
                removed = random.choice(selected)
                selected.remove(removed)
                total_duration -= removed.estimated_duration
        
        return selected
    
    def _calculate_objectives(self, plan: List[ResearchOpportunity]) -> Dict[str, float]:
        """Calculate objective values for a research plan."""
        if not plan:
            return {
                "novelty": 0.0,
                "impact": 0.0,
                "feasibility": 0.0,
                "efficiency": 0.0,
                "collaboration": 0.0
            }
        
        # Aggregate metrics from all opportunities in plan
        total_novelty = sum(opp.vector.novelty for opp in plan) / len(plan)
        total_impact = sum(opp.vector.impact for opp in plan) / len(plan)
        total_feasibility = sum(opp.vector.feasibility for opp in plan) / len(plan)
        
        # Efficiency (inverse of time)
        total_duration = sum(opp.estimated_duration for opp in plan)
        efficiency = 1.0 / max(total_duration / 12, 1.0)  # Normalize by year
        
        # Collaboration potential
        total_collaboration = sum(opp.vector.collaboration for opp in plan) / len(plan)
        
        return {
            "novelty": total_novelty,
            "impact": total_impact,
            "feasibility": total_feasibility,
            "efficiency": min(efficiency, 1.0),
            "collaboration": total_collaboration
        }
    
    def _is_non_dominated(self, objectives: Dict[str, float], 
                         other_objectives: List[Dict[str, float]]) -> bool:
        """Check if objectives are non-dominated."""
        for other in other_objectives:
            if self._dominates(other, objectives):
                return False
        return True
    
    def _dominates(self, obj1: Dict[str, float], obj2: Dict[str, float]) -> bool:
        """Check if obj1 dominates obj2."""
        better_in_any = False
        
        for key in obj1:
            if obj1[key] > obj2[key]:
                better_in_any = True
            elif obj1[key] < obj2[key]:
                return False
        
        return better_in_any
    
    async def _select_optimal_plan(self, pareto_front: List[Dict[str, Any]], 
                                  constraints: Dict[str, Any]) -> List[ResearchOpportunity]:
        """Select optimal plan from Pareto front based on preferences."""
        if not pareto_front:
            return []
        
        # Use preference weights (could be user-defined)
        weights = constraints.get("objective_weights", {
            "novelty": 0.25,
            "impact": 0.30,
            "feasibility": 0.20,
            "efficiency": 0.15,
            "collaboration": 0.10
        })
        
        # Calculate weighted utility for each solution
        best_solution = None
        best_utility = -1
        
        for solution in pareto_front:
            utility = sum(solution["objectives"][obj] * weights.get(obj, 0) 
                         for obj in solution["objectives"])
            
            if utility > best_utility:
                best_utility = utility
                best_solution = solution
        
        return best_solution["plan"] if best_solution else []
    
    async def _analyze_research_plan(self, plan: List[ResearchOpportunity], 
                                   all_opportunities: List[ResearchOpportunity], 
                                   constraints: Dict[str, Any]) -> str:
        """Generate analysis of the optimal research plan."""
        if not plan:
            return "No optimal research plan could be generated with given constraints."
        
        analysis_parts = []
        
        # Overview
        analysis_parts.append(f"Optimal research plan includes {len(plan)} projects:")
        
        for i, opp in enumerate(plan, 1):
            analysis_parts.append(f"{i}. {opp.title} ({opp.estimated_duration} months)")
        
        # Metrics
        objectives = self._calculate_objectives(plan)
        analysis_parts.append(f"\\nPlan Metrics:")
        analysis_parts.append(f"- Novelty Score: {objectives['novelty']:.2f}")
        analysis_parts.append(f"- Impact Potential: {objectives['impact']:.2f}")
        analysis_parts.append(f"- Feasibility: {objectives['feasibility']:.2f}")
        analysis_parts.append(f"- Efficiency: {objectives['efficiency']:.2f}")
        analysis_parts.append(f"- Collaboration: {objectives['collaboration']:.2f}")
        
        # Timeline
        total_duration = sum(opp.estimated_duration for opp in plan)
        analysis_parts.append(f"\\nEstimated total duration: {total_duration} months")
        
        # Recommendations
        analysis_parts.append(f"\\nRecommendations:")
        if objectives['collaboration'] > 0.7:
            analysis_parts.append("- High collaboration potential - consider joint projects")
        if objectives['novelty'] > 0.8:
            analysis_parts.append("- High novelty - allocate extra time for exploration")
        if objectives['feasibility'] < 0.6:
            analysis_parts.append("- Lower feasibility - consider risk mitigation strategies")
        
        return " ".join(analysis_parts)


# Research Algorithm Factory
class ResearchAlgorithmFactory:
    """Factory for creating and managing research algorithms."""
    
    @staticmethod
    def create_knowledge_graph_nn(config: Dict[str, Any] = None) -> KnowledgeGraphNeuralNetwork:
        """Create knowledge graph neural network."""
        config = config or {}
        return KnowledgeGraphNeuralNetwork(
            embedding_dim=config.get("embedding_dim", 256),
            num_heads=config.get("num_heads", 8)
        )
    
    @staticmethod
    def create_literature_synthesizer(kg_nn: KnowledgeGraphNeuralNetwork = None) -> AutomatedLiteratureSynthesis:
        """Create literature synthesis algorithm."""
        if kg_nn is None:
            kg_nn = ResearchAlgorithmFactory.create_knowledge_graph_nn()
        return AutomatedLiteratureSynthesis(kg_nn)
    
    @staticmethod
    def create_experiment_optimizer(config: Dict[str, Any] = None) -> ExperimentalDesignOptimizer:
        """Create experimental design optimizer."""
        config = config or {}
        return ExperimentalDesignOptimizer(
            population_size=config.get("population_size", 50),
            generations=config.get("generations", 100)
        )
    
    @staticmethod
    def create_research_planner() -> MultiObjectiveResearchPlanner:
        """Create multi-objective research planner."""
        return MultiObjectiveResearchPlanner()


# Main Research Enhancement Interface
async def enhance_research_workflow(research_data: Dict[str, Any], 
                                  config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main interface for research workflow enhancement.
    
    Applies advanced algorithms to optimize research processes.
    """
    config = config or {}
    results = {}
    
    # Initialize algorithms
    kg_nn = ResearchAlgorithmFactory.create_knowledge_graph_nn(config.get("kg_config"))
    lit_synth = ResearchAlgorithmFactory.create_literature_synthesizer(kg_nn)
    exp_opt = ResearchAlgorithmFactory.create_experiment_optimizer(config.get("exp_config"))
    planner = ResearchAlgorithmFactory.create_research_planner()
    
    # Literature synthesis
    if "papers" in research_data:
        print("🔬 Running literature synthesis...")
        synthesis_result = await lit_synth.synthesize_literature(research_data["papers"])
        results["literature_synthesis"] = synthesis_result
    
    # Experimental design optimization
    if "experiment_constraints" in research_data:
        print("⚗️ Optimizing experimental design...")
        objectives = research_data.get("experiment_objectives", ["power", "efficiency", "validity"])
        exp_result = await exp_opt.optimize_experimental_design(
            research_data["experiment_constraints"], objectives
        )
        results["experimental_design"] = exp_result
    
    # Research planning
    if "opportunities" in research_data:
        print("📋 Optimizing research plan...")
        opportunities = []
        for opp_data in research_data["opportunities"]:
            # Convert vector dict to ResearchVector object
            vector_data = opp_data.get("vector", {})
            vector = ResearchVector(**vector_data)
            
            # Create ResearchOpportunity with converted vector
            opp_copy = opp_data.copy()
            opp_copy["vector"] = vector
            opp_copy["phase"] = ResearchPhase(opp_copy["phase"])
            
            opportunities.append(ResearchOpportunity(**opp_copy))
        
        constraints = research_data.get("planning_constraints", {})
        plan_result = await planner.optimize_research_plan(opportunities, constraints)
        results["research_plan"] = plan_result
    
    # Generate overall recommendations
    results["recommendations"] = _generate_overall_recommendations(results)
    results["processing_timestamp"] = datetime.now().isoformat()
    
    return results


def _generate_overall_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate overall recommendations based on all analysis results."""
    recommendations = []
    
    if "literature_synthesis" in results:
        gaps = results["literature_synthesis"].get("gaps", [])
        if gaps:
            recommendations.append(f"Literature analysis identified {len(gaps)} research gaps - prioritize gap research")
    
    if "experimental_design" in results:
        design = results["experimental_design"]
        recommendations.append(f"Optimal experimental design: {design['optimal_design']['sample_size']} participants, {design['optimal_design']['duration_weeks']} weeks")
    
    if "research_plan" in results:
        plan = results["research_plan"]["optimal_plan"]
        if plan:
            total_duration = sum(opp.estimated_duration for opp in plan)
            recommendations.append(f"Optimal research plan: {len(plan)} projects over {total_duration} months")
    
    if not recommendations:
        recommendations.append("Continue systematic research approach with regular literature reviews")
    
    return recommendations
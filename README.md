# Self-Documenting PhD Notebook

A Python-based research notebook system for PhD students that manages Obsidian-compatible markdown vaults, auto-tags and links notes, and helps generate structured paper drafts.

Built for: **DocGraph dissertation** — multi-document knowledge graph construction pipeline
Institution: Towson University CS PhD | Advisor: Dr. Mike McGuire

---

## What It Does

| Feature | Description |
|---------|-------------|
| **Vault Management** | Creates and manages an Obsidian-compatible folder structure for research notes |
| **Note Types** | Tracks experiments, literature, ideas, meetings, projects, analyses, daily notes |
| **Knowledge Graph** | Builds a graph of connections between notes via `[[wiki-links]]` |
| **Auto-Tagging** | Keyword-based auto-tagging of notes (DocGraph, NLP, KG terms built-in) |
| **Smart Linking** | Suggests links between related notes by content similarity |
| **SPARCAgent** | Generates structured paper drafts (Intro/Problem/Method/Results/Conclusion) |
| **LiteratureAgent** | Creates structured literature review notes |
| **ExperimentAgent** | Tracks experiments with hypothesis, methodology, and timeline |
| **ArXiv Publisher** | Validates and formats submissions for arXiv (cs.IR, cs.AI, cs.CL categories) |
| **CLI (`sdpn`)** | Command-line interface for init, create, search, list, stats |

---

## Quick Start

```bash
# Install (uses conda Python 3.12)
cd ~/repos/Self-Documenting-PhD-Notebook
~/anaconda3/bin/pip install -e .

# Or use the convenience script
./run.sh setup

# Initialize your research vault
sdpn init DocGraph \
  --author "Daniel Schmidt" \
  --institution "Towson University" \
  --field "Computer Science" \
  --path ~/Documents/PhD_Research

# Open the vault in Obsidian
# File > Open Vault > ~/Documents/PhD_Research/DocGraph

# Create notes from CLI
sdpn create "DocGraph Architecture" --type project --tags "#docgraph,#kg"
sdpn list
sdpn search "knowledge graph"
sdpn stats
```

---

## DocGraph Dissertation Workflow

### 1. Add a paper to your literature collection

```python
from phd_notebook import ResearchNotebook, LiteratureAgent, SPARCAgent

nb = ResearchNotebook(
    vault_path="~/Documents/PhD_Research/DocGraph",
    author="Daniel Schmidt",
    institution="Towson University",
    field="Computer Science",
    subfield="Knowledge Graphs",
)

lit_agent = LiteratureAgent()
nb.register_agent(lit_agent)

paper = lit_agent.create_literature_note(
    "DocGraph: Cross-Document Knowledge Graph Construction",
    "Schmidt et al.",
    "2026",
)
paper.save(nb.vault_path / "literature" / f"{paper.title}.md")
```

### 2. Track an experiment

```python
from phd_notebook import ResearchNotebook
from phd_notebook.agents.experiment_agent import ExperimentAgent

exp_agent = ExperimentAgent()
nb.register_agent(exp_agent)

exp = exp_agent.create_experiment_note(
    "DocGraph vs Baseline on Multi-HopQA",
    hypothesis="DocGraph outperforms baselines by >5% F1 on multi-hop questions",
    experiment_type="comparative",
)
```

### 3. Generate an arXiv-ready paper draft

```python
from phd_notebook.agents.sparc_agent import SPARCAgent

sparc = SPARCAgent()
nb.register_agent(sparc)

draft = sparc.create_paper_draft(
    title="DocGraph: A Multi-Document Knowledge Graph Construction Pipeline",
    topic="knowledge graph construction from multiple documents",
    research_notes=[paper, exp],
    target_venue="EMNLP 2026",
    authors="Daniel Schmidt, Mike McGuire",
)

# Draft is now a Note with SPARC structure (Intro / Problem / Method / Results / Conclusion)
draft.save(nb.vault_path / "drafts" / "docgraph-arxiv.md")
```

### 4. Generate abstract

```python
abstract = sparc.generate_abstract(
    title="DocGraph",
    problem="constructing knowledge graphs from heterogeneous document collections",
    method="a pipeline combining entity resolution, cross-document coreference, and graph construction",
    results="our approach achieves state-of-the-art performance on multi-hop QA benchmarks",
)
```

### 5. Use the CLI

```bash
# Daily workflow
sdpn create "Meeting with Dr. McGuire" --type meeting --tags "#meeting,#advisor"
sdpn search "DocGraph"
sdpn list --type experiment
sdpn stats
```

---

## Project Structure

```
Self-Documenting-PhD-Notebook/
├── src/phd_notebook/
│   ├── core/
│   │   ├── note.py          # Note class with YAML frontmatter + wiki-links
│   │   ├── notebook.py      # ResearchNotebook - main interface
│   │   ├── vault_manager.py # Obsidian vault CRUD
│   │   └── knowledge_graph.py # Note graph construction
│   ├── agents/
│   │   ├── base.py          # BaseAgent
│   │   ├── literature_agent.py  # Structured literature notes
│   │   ├── experiment_agent.py  # Experiment tracking
│   │   ├── sparc_agent.py   # Paper draft generation (SPARC methodology)
│   │   └── smart_agent.py   # Auto-tagging + link suggestions
│   ├── ai/
│   │   ├── base_ai.py       # BaseAI + MockAI
│   │   ├── anthropic_client.py
│   │   ├── openai_client.py
│   │   └── client_factory.py
│   ├── publication/
│   │   ├── arxiv_publisher.py  # arXiv submission validation & formatting
│   │   ├── citation_manager.py
│   │   ├── journal_publisher.py
│   │   └── latex_compiler.py
│   ├── workflows/
│   │   ├── automation.py    # Auto-tagging, smart linking, daily review
│   │   └── daily_workflows.py
│   ├── performance/         # Caching, async processing, search index
│   ├── monitoring/          # Logging setup
│   ├── research/            # Research tracker, hypothesis engine, publication pipeline
│   └── cli/
│       └── main.py          # `sdpn` CLI entry point
├── tests/
│   ├── unit/                # Unit tests (110 passing)
│   └── integration/         # Integration tests
├── run.sh                   # Quick setup/run script
├── requirements.txt
└── pyproject.toml
```

---

## Configuration

The notebook supports optional AI backends (set env vars for real LLM use):

```bash
# Optional: real AI for smarter tagging/analysis
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Without these, the system runs with MockAI (keyword-based, no API calls)
```

---

## Development

```bash
# Run tests
~/anaconda3/bin/python3 -m pytest tests/ -q

# Run specific test file
~/anaconda3/bin/python3 -m pytest tests/unit/test_note.py -v

# Install in dev mode
~/anaconda3/bin/pip install -e .
```

---

## Connecting to the DocGraph Repo

```python
# Cross-reference notes with your actual codebase
from pathlib import Path

docgraph_path = Path("~/repos/docgraph-publication").expanduser()
vault_path = Path("~/Documents/PhD_Research/DocGraph").expanduser()

# When you add a new module to DocGraph, create a matching note:
note = nb.create_note(
    "DocGraph Entity Resolution Module",
    content=f"Implementation at: `{docgraph_path}/src/entity_resolution.py`\n\n"
            "[[DocGraph Architecture]] [[Cross-Document Coreference]]",
    note_type=NoteType.PROJECT,
    tags=["#docgraph", "#implementation"],
)
```

---

## License

MIT — see [LICENSE](LICENSE)

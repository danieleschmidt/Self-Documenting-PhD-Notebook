#!/usr/bin/env bash
# run.sh - Quick setup and usage script for the PhD Notebook
# Uses ~/anaconda3/bin/python3 (3.12)

set -e

PYTHON="${PYTHON:-$HOME/anaconda3/bin/python3}"
PIP="${PIP:-$HOME/anaconda3/bin/pip}"

case "${1:-help}" in
  setup)
    echo "📦 Installing Self-Documenting PhD Notebook..."
    $PIP install -e . -q
    echo "✅ Installed. Run 'sdpn --help' to get started."
    ;;

  test)
    echo "🧪 Running tests..."
    $PYTHON -m pytest tests/ -q "${@:2}"
    ;;

  init)
    # Quick init for DocGraph vault
    VAULT="${2:-$HOME/Documents/PhD_Research/DocGraph}"
    echo "🚀 Initializing DocGraph vault at $VAULT"
    sdpn init DocGraph \
      --author "Daniel Schmidt" \
      --institution "Towson University" \
      --field "Computer Science" \
      --path "$(dirname $VAULT)"
    ;;

  demo)
    echo "🔬 Running quick demo..."
    $PYTHON - <<'EOF'
import sys, os
sys.path.insert(0, 'src')
import tempfile
from pathlib import Path
from phd_notebook import ResearchNotebook
from phd_notebook.agents.sparc_agent import SPARCAgent
from phd_notebook.agents.experiment_agent import ExperimentAgent
from phd_notebook.core.note import NoteType

with tempfile.TemporaryDirectory() as tmp:
    nb = ResearchNotebook(
        vault_path=tmp + "/DocGraph",
        author="Daniel Schmidt",
        institution="Towson University",
        field="Computer Science",
        subfield="Knowledge Graphs",
    )

    # Register agents
    sparc = SPARCAgent()
    exp_agent = ExperimentAgent()
    nb.register_agent(sparc)
    nb.register_agent(exp_agent)

    # Create experiment note
    exp = exp_agent.create_experiment_note(
        "DocGraph vs Baseline on Multi-HopQA",
        hypothesis="DocGraph outperforms baselines by >5% F1",
        experiment_type="comparative",
    )
    print(f"\n✅ Created experiment: {exp.title}")

    # Generate paper draft
    draft = sparc.create_paper_draft(
        title="DocGraph: Multi-Document Knowledge Graph Construction",
        topic="knowledge graph construction from heterogeneous documents",
        research_notes=[exp],
        target_venue="EMNLP 2026",
        authors="Daniel Schmidt, Mike McGuire",
    )
    print(f"✅ Generated paper draft: {draft.title}")
    print(f"   Tags: {draft.frontmatter.tags}")
    print(f"\n📄 Draft outline (first 500 chars):")
    print(draft.content[:500])

    print(f"\n📊 Vault stats: {nb.get_stats()}")
EOF
    ;;

  help|*)
    echo "Self-Documenting PhD Notebook"
    echo ""
    echo "Usage: ./run.sh <command>"
    echo ""
    echo "Commands:"
    echo "  setup    Install the package (pip install -e .)"
    echo "  test     Run all tests"
    echo "  init     Initialize DocGraph vault (default: ~/Documents/PhD_Research/DocGraph)"
    echo "  demo     Run a quick demo showing core functionality"
    echo "  help     Show this help"
    echo ""
    echo "After setup, use the 'sdpn' CLI:"
    echo "  sdpn init DocGraph --author 'Daniel Schmidt' --institution 'Towson University' --field 'Computer Science'"
    echo "  sdpn create 'Meeting with Dr. McGuire' --type meeting"
    echo "  sdpn search 'knowledge graph'"
    echo "  sdpn list --type experiment"
    echo "  sdpn stats"
    ;;
esac

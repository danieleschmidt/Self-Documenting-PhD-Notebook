# Self-Documenting PhD Notebook - Development Makefile

.PHONY: help install install-dev test test-unit test-integration lint format security-check type-check docs clean build publish

# Default target
help: ## Show this help message
	@echo "Self-Documenting PhD Notebook - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install package for production use
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,all]"
	pre-commit install

install-minimal: ## Install with minimal dependencies only
	pip install -e .

# Testing
test: ## Run all tests with coverage
	pytest tests/ -v --cov=src/phd_notebook --cov-report=term-missing --cov-report=html

test-unit: ## Run only unit tests
	pytest tests/unit/ -v --cov=src/phd_notebook

test-integration: ## Run only integration tests
	pytest tests/integration/ -v

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v --no-cov

test-watch: ## Run tests in watch mode
	pytest-watch tests/ -- -v

# Code Quality
lint: ## Run linting checks
	flake8 src/ tests/
	isort --check-only --diff src/ tests/
	black --check src/ tests/

format: ## Auto-format code
	isort src/ tests/
	black src/ tests/

type-check: ## Run type checking
	mypy src/phd_notebook/

security-check: ## Run security checks
	bandit -r src/ -ll
	safety check --json
	pip-audit

pre-commit-all: ## Run all pre-commit hooks
	pre-commit run --all-files

# Performance & Benchmarks
benchmark: ## Run performance benchmarks
	python -m pytest tests/benchmarks/ -v --benchmark-only

profile: ## Profile application performance
	python -m cProfile -o profile.stats -m phd_notebook.cli --help
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('tottime').print_stats(20)"

# Documentation
docs: ## Generate documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs && make html && python -m http.server 8000 -d _build/html

docs-clean: ## Clean documentation build
	cd docs && make clean

# Database/Cache
reset-cache: ## Clear all caches
	find . -name "*.cache" -delete
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Build & Release
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

build: clean ## Build distribution packages
	python -m build

check-build: build ## Check build artifacts
	python -m twine check dist/*

publish-test: build ## Publish to test PyPI
	python -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	python -m twine upload dist/*

# Development Utilities
init-vault: ## Initialize a test vault for development
	python -c "from phd_notebook import ResearchNotebook; ResearchNotebook('./dev_vault', author='Developer', field='Testing')"

demo: ## Run demonstration workflow
	python scripts/demo.py

validate-vault: ## Validate an existing vault
	python -c "from phd_notebook.utils.validation import DataIntegrityChecker; import sys; checker = DataIntegrityChecker('./dev_vault' if len(sys.argv) == 1 else sys.argv[1]); results = checker.run_full_check(); print(f'Health: {results[\"overall_health\"]}, Errors: {results[\"total_errors\"]}, Warnings: {results[\"total_warnings\"]}')"

# CI/CD Simulation
ci-test: ## Run full CI test suite
	@echo "🚀 Running CI Test Suite..."
	make install-dev
	make lint
	make type-check
	make security-check
	make test
	@echo "✅ CI Tests Passed!"

ci-build: ## Run full CI build process
	@echo "🏗️ Running CI Build Process..."
	make clean
	make ci-test
	make build
	make check-build
	@echo "✅ CI Build Complete!"

# Performance Validation
perf-test: ## Run performance validation
	@echo "⚡ Running Performance Tests..."
	python -m pytest tests/performance/ -v
	@echo "📊 Performance validation complete"

load-test: ## Run load testing
	@echo "🔄 Running Load Tests..."
	python scripts/load_test.py
	@echo "📈 Load testing complete"

# Security Validation
security-full: ## Run comprehensive security checks
	@echo "🔒 Running Security Validation..."
	bandit -r src/ -f json -o security-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=audit-report.json
	@echo "🛡️ Security validation complete"

# Monitoring & Health
health-check: ## Check system health
	@echo "🏥 System Health Check..."
	python -c "import phd_notebook; print('✅ Package imports successfully')"
	python -c "from phd_notebook.utils.validation import validate_note_data; validate_note_data({'title': 'Test', 'note_type': 'idea'}); print('✅ Validation working')"
	python -c "from phd_notebook.cache.memory_cache import MemoryCache; cache = MemoryCache(); cache.set('test', 'value'); assert cache.get('test') == 'value'; print('✅ Caching working')"
	@echo "💚 System health check passed!"

# Development Server (for future web interface)
dev-server: ## Start development server (placeholder)
	@echo "🌐 Development server would start here..."
	@echo "This is a placeholder for future web interface"

# Release Management
version-bump-patch: ## Bump patch version
	bump2version patch

version-bump-minor: ## Bump minor version
	bump2version minor

version-bump-major: ## Bump major version
	bump2version major

changelog: ## Generate changelog
	@echo "📝 Changelog would be generated here"
	@echo "Future: Auto-generate from commit messages"

# Quality Gates (for CI)
quality-gate-security: ## Security quality gate
	@echo "🔐 Security Quality Gate"
	bandit -r src/ -ll -f json | jq '.results | length' | xargs -I {} test {} -eq 0
	safety check
	@echo "✅ Security gate passed"

quality-gate-performance: ## Performance quality gate  
	@echo "⚡ Performance Quality Gate"
	python -m pytest tests/performance/ --benchmark-min-time=0.1
	@echo "✅ Performance gate passed"

quality-gate-coverage: ## Coverage quality gate
	@echo "📊 Coverage Quality Gate"
	pytest tests/ --cov=src/phd_notebook --cov-fail-under=85 --quiet
	@echo "✅ Coverage gate passed"

all-quality-gates: quality-gate-security quality-gate-performance quality-gate-coverage ## Run all quality gates
	@echo "🎯 All quality gates passed!"

# Installation Verification
verify-install: ## Verify installation works correctly
	@echo "🔍 Installation Verification..."
	python -c "import phd_notebook; print(f'Version: {phd_notebook.__version__}')"
	python -c "from phd_notebook import ResearchNotebook; print('✅ Core imports work')"
	python -c "from phd_notebook.cli.main import main; print('✅ CLI available')"
	which sdpn || echo "⚠️ CLI not in PATH"
	@echo "✅ Installation verification complete"
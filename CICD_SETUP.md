# CI/CD Pipeline Setup Guide

Due to GitHub App permissions, the CI/CD workflow file needs to be created manually. Here's the complete configuration:

## GitHub Actions Workflow

Create `.github/workflows/ci.yml` in your repository with the following content:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: "3.9"

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,all]"
    
    - name: Lint with flake8
      run: |
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Format check with black
      run: black --check src/ tests/
    
    - name: Import order check with isort
      run: isort --check-only --diff src/ tests/
    
    - name: Type check with mypy
      run: mypy src/phd_notebook/
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=src/phd_notebook --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit[toml] safety pip-audit
        pip install -e .
    
    - name: Run bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json
    
    - name: Run safety check
      run: safety check --json --output safety-report.json
    
    - name: Run pip audit
      run: pip-audit --format=json --output=audit-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          audit-report.json

  performance:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,all]"
        pip install pytest-benchmark
    
    - name: Run performance benchmarks
      run: |
        python -c "
        from phd_notebook.cache.memory_cache import MemoryCache
        import time
        
        # Basic performance test
        cache = MemoryCache(max_size=10000)
        start = time.time()
        for i in range(1000):
            cache.set(f'key_{i}', f'value_{i}')
        for i in range(1000):
            cache.get(f'key_{i}')
        duration = time.time() - start
        print(f'Cache operations: {duration:.3f}s for 2000 ops')
        assert duration < 1.0, f'Performance regression: {duration}s > 1.0s'
        "
    
    - name: Memory usage test
      run: |
        python -c "
        from phd_notebook import ResearchNotebook
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Create notebook and add content
        notebook = ResearchNotebook('/tmp/perf_test', author='Test')
        for i in range(100):
            notebook.create_note(f'Note {i}', f'Content for note {i}' * 100)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        print(f'Memory growth: {memory_growth:.1f} MB')
        assert memory_growth < 100, f'Memory usage too high: {memory_growth}MB'
        "

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: python -m twine check dist/*
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: build
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: dist
        path: dist/
    
    - name: Install from wheel
      run: |
        pip install dist/*.whl
    
    - name: Test installation
      run: |
        python -c "import phd_notebook; print(f'Installed version: {phd_notebook.__version__}')"
        which sdpn
        sdpn --help
    
    - name: Test CLI functionality
      run: |
        mkdir /tmp/test_vault_integration
        cd /tmp/test_vault_integration
        echo "Test Author" | sdpn init "Test Vault" --path . --field "Computer Science"
        sdpn create "Test Note" --type idea --tags test,integration
        sdpn list
        sdpn search "test"
        sdpn stats

  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: [test, security, build, integration-test]
    if: github.event_name == 'release' && github.event.action == 'published'
    
    steps:
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: dist
        path: dist/
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}

  docker:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-docs:
    name: Deploy Documentation
    runs-on: ubuntu-latest
    needs: [test]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install sphinx sphinx-rtd-theme
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## Setup Instructions

1. **Create the workflow file manually**:
   - Go to your GitHub repository
   - Navigate to `.github/workflows/`
   - Create a new file named `ci.yml`
   - Copy the above content into the file
   - Commit the file

2. **Configure repository secrets** (if needed):
   - `PYPI_API_TOKEN`: For publishing to PyPI
   - `CODECOV_TOKEN`: For code coverage reporting

3. **Enable GitHub Pages** (for documentation):
   - Go to repository Settings > Pages
   - Set source to "Deploy from a branch"
   - Select the `gh-pages` branch

## Manual CI/CD Commands

You can run the same checks locally using:

```bash
# Run all quality gates
make ci-test

# Individual commands
make test           # Run tests with coverage
make lint          # Code linting
make type-check    # Type checking
make security-check # Security scanning
make build         # Build package
```

This setup provides comprehensive CI/CD coverage including testing, security scanning, performance benchmarks, and automated deployment.
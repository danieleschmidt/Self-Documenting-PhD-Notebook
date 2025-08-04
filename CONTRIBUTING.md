# Contributing to Self-Documenting PhD Notebook 🎓

Thank you for your interest in contributing to the Self-Documenting PhD Notebook! This project aims to revolutionize academic research workflows through AI-powered automation and intelligent organization.

## 🌟 Ways to Contribute

### 1. Research Templates & Workflows
- **Field-Specific Templates**: Create templates for your research domain
- **Automation Workflows**: Share successful research automation patterns
- **Best Practices**: Document effective PhD management strategies

### 2. AI Agents Development
- **Domain-Specific Agents**: Develop agents for specialized research areas
- **Integration Agents**: Create agents that connect to research tools
- **Analysis Agents**: Build agents for data analysis and interpretation

### 3. Data Connectors
- **Lab Equipment**: Integrate with scientific instruments
- **Cloud Services**: Connect to research data platforms
- **Academic Tools**: Interface with reference managers, databases

### 4. Core Development
- **Bug Fixes**: Help improve stability and reliability
- **Performance**: Optimize caching, indexing, and search
- **Features**: Implement new core functionality

## 🚀 Getting Started

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/danieleschmidt/Self-Documenting-PhD-Notebook.git
   cd Self-Documenting-PhD-Notebook
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   make install-dev
   ```

3. **Run initial tests**
   ```bash
   make test
   make lint
   make type-check
   ```

4. **Create a development vault**
   ```bash
   make init-vault
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   make test
   
   # Run specific test categories
   make test-unit
   make test-integration
   
   # Check code quality
   make lint
   make type-check
   make security-check
   ```

4. **Commit your changes**
   ```bash
   # Format code first
   make format
   
   # Commit with descriptive message
   git add .
   git commit -m "feat(agents): add literature review agent for CS papers"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Code Style
- **Formatting**: Use `black` for code formatting (88 character line length)
- **Imports**: Use `isort` for import organization
- **Linting**: Follow `flake8` guidelines
- **Type Hints**: Use type hints for all function signatures

### Example Code Structure
```python
"""
Module docstring describing the purpose and functionality.
"""

from typing import Dict, List, Optional
from datetime import datetime

from ..core.note import Note
from ..utils.logging import get_logger


class ExampleAgent:
    """
    Brief description of the class.
    
    Longer description with usage examples if needed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
    
    def process_data(self, data: str, options: Optional[Dict] = None) -> str:
        """
        Process input data and return result.
        
        Args:
            data: Input data to process
            options: Optional processing configuration
            
        Returns:
            Processed data string
            
        Raises:
            ValueError: If data is invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Implementation here
        return processed_data
```

### Testing Standards
- **Coverage**: Maintain >85% test coverage
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows
- **Test Structure**: Use descriptive test names and clear assertions

```python
"""
Test module for ExampleAgent.
"""

import pytest
from phd_notebook.agents.example import ExampleAgent


class TestExampleAgent:
    """Test cases for ExampleAgent class."""
    
    def test_process_data_success(self):
        """Test successful data processing."""
        agent = ExampleAgent({'setting': 'value'})
        result = agent.process_data('test input')
        
        assert result is not None
        assert 'processed' in result.lower()
    
    def test_process_data_empty_input(self):
        """Test that empty input raises ValueError."""
        agent = ExampleAgent({})
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            agent.process_data('')
```

## 🏗️ Architecture Guidelines

### Core Components
- **Notes**: Central data structure for all content
- **Vault Manager**: File system operations and organization
- **Knowledge Graph**: Relationship tracking and analysis
- **Agents**: AI-powered automation and analysis
- **Connectors**: External data source integrations

### Design Principles
1. **Modularity**: Components should be loosely coupled
2. **Extensibility**: Easy to add new agents and connectors
3. **Performance**: Efficient caching and indexing
4. **Security**: Input validation and safe file operations
5. **Reliability**: Comprehensive error handling and logging

### Adding New Components

#### New AI Agent
1. Inherit from `BaseAgent`
2. Implement required methods
3. Add comprehensive tests
4. Document capabilities and usage

```python
from ..agents.base import BaseAgent

class MyResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="my_research_agent",
            capabilities=["data_analysis", "report_generation"]
        )
    
    def process(self, input_data, **kwargs):
        # Implementation here
        return processed_result
```

#### New Data Connector
1. Inherit from `DataConnector`
2. Implement connection and data fetching
3. Handle authentication and errors
4. Add configuration options

```python
from ..connectors.base import DataConnector

class MyDataConnector(DataConnector):
    def connect(self):
        # Establish connection
        return True
    
    def fetch_data(self, **kwargs):
        # Yield data items
        yield {'title': 'Data Item', 'content': 'Content'}
```

## 🧪 Testing Guidelines

### Test Categories
- **Unit Tests** (`tests/unit/`): Test individual components
- **Integration Tests** (`tests/integration/`): Test component interactions
- **Performance Tests** (`tests/performance/`): Benchmark critical operations
- **Security Tests** (`tests/security/`): Validate security measures

### Running Tests
```bash
# All tests with coverage
make test

# Specific categories
make test-unit
make test-integration

# Performance benchmarks
make benchmark

# Security validation
make security-check
```

### Test Data
- Use fixtures for reusable test data
- Create temporary directories for file operations
- Mock external services and APIs
- Use realistic but safe test data

## 📚 Documentation

### Required Documentation
1. **Docstrings**: All public functions and classes
2. **Type Hints**: Function signatures and complex types
3. **README Updates**: New features and usage examples
4. **API Documentation**: For new public interfaces

### Documentation Style
- Use Google-style docstrings
- Include usage examples for complex functionality
- Document all parameters and return values
- Add cross-references to related functionality

## 🔒 Security Considerations

### Input Validation
- Sanitize all user inputs
- Validate file paths for security
- Check for malicious content in notes
- Use parameterized queries for databases

### Data Protection
- Never log sensitive information
- Implement proper access controls
- Use secure defaults for configurations
- Follow OWASP security guidelines

### Security Testing
```bash
# Run security scans
make security-check

# Check for known vulnerabilities
make security-full
```

## 🎯 Performance Guidelines

### Optimization Priorities
1. **Memory Usage**: Efficient caching and cleanup
2. **I/O Operations**: Minimize file system access
3. **Search Performance**: Fast indexing and retrieval
4. **Startup Time**: Quick initialization

### Performance Testing
```bash
# Run benchmarks
make benchmark

# Profile performance
make profile

# Load testing
make load-test
```

## 📋 Pull Request Process

### Before Submitting
1. ✅ All tests pass (`make test`)
2. ✅ Code follows style guidelines (`make lint`)
3. ✅ Type checking passes (`make type-check`)
4. ✅ Security scans pass (`make security-check`)
5. ✅ Documentation is updated
6. ✅ Performance is acceptable

### Pull Request Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new security vulnerabilities
- [ ] Performance impact assessed
```

### Review Process
1. **Automated Checks**: CI/CD pipeline runs all tests
2. **Code Review**: Maintainer reviews code quality
3. **Testing**: Functionality testing in clean environment
4. **Documentation**: Verify documentation accuracy
5. **Merge**: Approved changes merged to main branch

## 🌍 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Celebrate diverse perspectives

### Communication Channels
- **Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Pull Requests**: Code contributions
- **Discord**: Real-time community chat (future)

### Recognition
Contributors are recognized in:
- Release notes for significant contributions
- README contributor section
- Annual contributor highlights

## 🎓 Research Domain Contributions

### Field-Specific Contributions
We especially welcome contributions from researchers in:

- **Computer Science**: ML/AI tools and workflows
- **Life Sciences**: Lab data integration and analysis
- **Physics**: Simulation data and instrument connectivity
- **Social Sciences**: Survey data and statistical analysis
- **Engineering**: Design documentation and testing workflows
- **Mathematics**: Proof organization and theorem tracking

### Template Contributions
Create templates for:
- Experiment protocols
- Literature review formats
- Data analysis workflows
- Thesis chapter structures
- Conference paper outlines

### Workflow Automation
Share automation for:
- Daily research logging
- Literature monitoring
- Data backup procedures
- Progress tracking
- Collaboration management

## 📞 Getting Help

### Resources
- **Documentation**: [docs.phd-notebook.org](https://docs.phd-notebook.org)
- **Examples**: Check `examples/` directory
- **Issues**: Search existing GitHub issues
- **Discussions**: GitHub Discussions for questions

### Mentorship
New contributors can request mentorship for:
- First-time contributions
- Architecture understanding
- Testing best practices
- Domain-specific implementations

## 🏆 Contributor Levels

### Contributor
- Submit bug reports
- Create feature requests
- Contribute templates and workflows
- Help with documentation

### Regular Contributor
- Multiple merged pull requests
- Help review other contributions
- Participate in project discussions
- Mentor new contributors

### Core Contributor
- Significant architectural contributions
- Maintain project components
- Guide project direction
- Lead feature development

### Maintainer
- Merge privileges
- Release management
- Project governance
- Community leadership

## 📅 Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Schedule
- **Patch releases**: As needed for critical fixes
- **Minor releases**: Monthly feature releases
- **Major releases**: Quarterly with breaking changes

Thank you for contributing to the future of academic research! 🚀

---

*Questions? Start a discussion or reach out to maintainers.*
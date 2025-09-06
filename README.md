# RL-Based API Test Suite Generator

🤖 **Reinforcement Learning** + 🔗 **API Dependencies** + 📋 **Postman Collections**

An advanced system that uses reinforcement learning to automatically generate comprehensive REST API test suites from OpenAPI specifications. The system discovers inter-service dependencies, verifies them through API calls, and uncovers stateful bugs arising from improper operation sequences.

## 🌟 Features

- **🔍 Smart Dependency Analysis**: RESTler-inspired grammar-based analysis to identify data flows between microservices
- **🤖 RL-Powered Testing**: PPO-based agent learns optimal test sequences through API interactions
- **🧠 LLM Enhancement**: Transformer models boost hypothesis generation from endpoint descriptions
- **📋 Postman Integration**: Export test suites as Postman collections with chained requests and assertions
- **🐛 Bug Discovery**: Automatically detect stateful bugs, race conditions, and invalid sequences
- **📊 Interactive CLI**: User-friendly command-line interface with colored output and progress bars
- **🔄 Feedback Loops**: Interactive refinement of test sequences based on user feedback

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RL API Test Suite Generator                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐ │
│  │    CLI      │───▶│   Spec Parser    │───▶│     Dependency Analyzer     │ │
│  │ (cli.py)    │    │ (spec_parser.py) │    │  (dependency_analyzer.py)   │ │
│  └─────────────┘    └──────────────────┘    └─────────────────────────────┘ │
│         │                     │                           │                 │
│         │                     ▼                           ▼                 │
│         │            ┌─────────────────┐         ┌──────────────────┐       │
│         │            │ LLM Integration │◀────────┤   RL Environment │       │
│         │            │ (transformers)  │         │   (gymnasium)    │       │
│         │            └─────────────────┘         └──────────────────┘       │
│         │                     │                           │                 │
│         │                     ▼                           ▼                 │
│         │            ┌─────────────────┐         ┌──────────────────┐       │
│         │            │Enhanced Hypoths │         │   PPO Agent      │       │
│         │            │   Generation    │         │ (stable-baselines3)│     │
│         │            └─────────────────┘         └──────────────────┘       │
│         │                                                 │                 │
│         ▼                                                 ▼                 │
│  ┌─────────────────┐                              ┌──────────────────┐       │
│  │ Postman Export  │◀─────────────────────────────┤  API Testing     │       │
│  │(postman_gen.py) │                              │   Execution      │       │
│  └─────────────────┘                              │   (requests)     │       │
│         │                                         └──────────────────┘       │
│         ▼                                                                   │
│  ┌─────────────────┐                                                       │
│  │   Collections   │                                                       │
│  │   (.json)       │                                                       │
│  └─────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Running microservices with OpenAPI documentation
- At least 4GB RAM (for LLM features)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rl-api-test-generator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Test installation** (optional):
```bash
python test_installation.py
```

4. **Fix common issues** (if needed):
```bash
# If you get dependency errors (TensorBoard, Rich, etc.)
python fix_dependencies.py

# Or install manually
pip install tensorboard rich tqdm
```

5. **Run the generator**:
```bash
# With running services
python main.py generate \
  http://localhost:8060/employee/v3/api-docs \
  http://localhost:8060/department/v3/api-docs \
  http://localhost:8060/organization/v3/api-docs

# Or test with sample spec file
python main.py generate examples/sample_openapi.yaml
```

## 📖 Usage Examples

### 1. Basic Usage

Generate test suite from OpenAPI specs:

```bash
# From URLs
python main.py generate \
  http://localhost:8060/employee/v3/api-docs \
  http://localhost:8060/department/v3/api-docs \
  --output my_test_suite.json

# From local files
python main.py generate specs/*.yaml --output local_tests.json
```

### 2. Advanced Configuration

```bash
python main.py generate \
  http://localhost:8060/employee/v3/api-docs \
  http://localhost:8060/department/v3/api-docs \
  --base-url http://localhost:8060 \
  --training-steps 20000 \
  --num-sequences 10 \
  --max-sequence-length 30 \
  --collection-name "Microservices Integration Tests" \
  --export-graph \
  --interactive
```

### 3. Interactive Mode

Enable interactive mode for real-time feedback and refinement:

```bash
python main.py generate specs/*.yaml --interactive
```

Features in interactive mode:
- Real-time sequence quality assessment
- User feedback collection
- Dynamic sequence refinement
- Manual sequence filtering

### 4. Dependency Analysis Only

Analyze dependencies without generating test suites:

```bash
python main.py analyze \
  http://localhost:8060/employee/v3/api-docs \
  http://localhost:8060/department/v3/api-docs \
  --output dependency_analysis.json \
  --graph-output dependency_graph.dot
```

### 5. Validate Generated Collections

```bash
python main.py validate my_test_suite.json
```

## 🔧 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output` | Output Postman collection file | `api_test_collection.json` |
| `--base-url` | Base URL for API services | `http://localhost:8060` |
| `--training-steps` | Number of RL training steps | `10000` |
| `--num-sequences` | Number of test sequences to generate | `5` |
| `--max-sequence-length` | Maximum length of test sequences | `20` |
| `--collection-name` | Name for Postman collection | `RL Generated API Tests` |
| `--export-graph` | Export dependency graph to DOT file | `False` |
| `--interactive` | Enable interactive mode | `False` |
| `--verbose` | Enable verbose logging | `False` |

## 🎯 Reward System

The RL agent uses a sophisticated reward system to learn optimal test sequences:

| Event | Reward | Description |
|-------|--------|-------------|
| **Verified Dependency** | +10 | Successful API call confirming hypothesized dependency |
| **Bug Discovery** | +20 | Uncovered stateful bug (race conditions, invalid sequences) |
| **Expected Failure** | +5 | 4xx error confirming missing dependency |
| **Successful Call** | +5 | Basic reward for successful API interactions |
| **Server Error** | +15 | 5xx error indicating potential server-side issues |
| **Unexpected Failure** | -5 | Network errors or unexpected failures |
| **Inefficient Action** | -1 | Redundant or non-productive API calls |

## 📊 Output Formats

### Postman Collection

Generated collections include:
- **Chained Requests**: Automatic variable extraction and chaining
- **Test Assertions**: Status code, response time, and content validation
- **Environment Variables**: Dynamic parameter generation
- **Pre-request Scripts**: Setup and data preparation
- **Bug Documentation**: Annotated requests that discovered issues

*Note: Collections are generated as pure JSON without external dependencies, ensuring maximum compatibility.*

### Dependency Graph

Export dependency graphs in Graphviz DOT format:
```bash
dot -Tpng dependency_graph.dot -o dependencies.png
```

### Analysis Reports

JSON reports containing:
- Service specifications summary
- Dependency hypothesis details
- Confidence scores and evidence
- Bug discovery statistics

## 🧠 AI Components

### 1. Dependency Analyzer

Uses RESTler-inspired grammar-based analysis:
- **Data Flow Detection**: Matches output fields to input parameters
- **Sequence Analysis**: Identifies CRUD operation dependencies
- **Schema Similarity**: Compares request/response structures
- **Resource Hierarchy**: Detects parent-child relationships

### 2. RL Agent (PPO)

Reinforcement learning environment:
- **State Space**: Current graph, API responses, sequence context
- **Action Space**: Next endpoint to call + parameter variations
- **Policy Network**: Multi-layer perceptron with attention mechanism
- **Training**: PPO algorithm with custom reward shaping

### 3. LLM Integration

Transformer-based enhancement:
- **Semantic Analysis**: Extract business logic from descriptions
- **Hypothesis Generation**: Suggest dependencies based on context
- **Parameter Intelligence**: Generate realistic test data
- **Bug Pattern Recognition**: Identify potential failure scenarios

## 🐛 Bug Discovery Capabilities

The system automatically detects:

1. **Race Conditions**: Concurrent access issues (409 Conflict responses)
2. **State Inconsistencies**: Invalid operation sequences
3. **Missing Dependencies**: 4xx errors indicating prerequisite failures
4. **Resource Leaks**: Uncleaned resources after failed operations
5. **Authentication Issues**: Token expiration and permission problems
6. **Data Integrity**: Inconsistent responses across related endpoints

## 🔍 Example Scenarios

### Microservices Integration Testing

For a typical microservices architecture:

```
Employee Service ──┐
                   ├──▶ Organization Service
Department Service ─┘
```

The system discovers:
1. **Data Dependencies**: Employee creation returns ID used by Department endpoints
2. **Sequence Requirements**: Organization must exist before Department creation
3. **Auth Dependencies**: All services require tokens from auth endpoints
4. **Resource Hierarchies**: Departments belong to Organizations

### Generated Test Sequence Example

```
1. POST /auth/login → Extract auth token
2. POST /organizations → Extract organization ID
3. POST /departments → Use organization ID, extract department ID  
4. POST /employees → Use department ID, extract employee ID
5. GET /employees/{id} → Verify employee creation
6. PUT /employees/{id} → Test employee updates
7. DELETE /employees/{id} → Test cleanup sequence
```

## 🛠️ Development

### Project Structure

```
rl-api-test-generator/
├── src/
│   ├── spec_parser.py          # OpenAPI spec parsing and validation
│   ├── dependency_analyzer.py  # Hypothesis graph building
│   ├── rl_agent.py            # PPO-based RL environment
│   ├── postman_generator.py   # Postman collection export
│   ├── llm_integration.py     # LLM-enhanced analysis
│   └── cli.py                 # Interactive command-line interface
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── ARCHITECTURE.md           # Detailed architecture documentation
```

### Running Tests

```bash
# Run basic functionality test
python -m src.spec_parser example_specs/petstore.yaml

# Test dependency analysis
python -m src.dependency_analyzer example_specs/*.yaml

# Test RL training (minimal)
python -m src.rl_agent example_specs/petstore.yaml
```

### Adding New Features

1. **Custom Reward Functions**: Modify `_calculate_reward()` in `rl_agent.py`
2. **New Dependency Types**: Extend `dependency_patterns` in `dependency_analyzer.py`
3. **LLM Models**: Update model configuration in `llm_integration.py`
4. **Export Formats**: Add new generators in `postman_generator.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RESTler**: Inspiration for grammar-based API analysis
- **Stable-Baselines3**: RL algorithm implementations
- **OpenAPI Initiative**: Specification standards
- **Postman**: Collection format and testing platform
- **NetworkX**: Graph analysis and visualization
- **Transformers**: LLM integration for enhanced analysis

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation in `ARCHITECTURE.md`
- Review example configurations in the `examples/` directory

---

**🎉 Happy API Testing!** Generate smarter test suites with the power of reinforcement learning! 
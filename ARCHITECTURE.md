# RL-Based API Test Suite Generator Architecture

## System Overview

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
│         │            │  OpenAPI Specs  │         │ Hypothesis Graph │       │
│         │            │   Validation    │         │   (NetworkX)     │       │
│         │            └─────────────────┘         └──────────────────┘       │
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
│         ▼                                                 │                 │
│  ┌─────────────────┐                                     ▼                 │
│  │   Collections   │                              ┌──────────────────┐       │
│  │   (.json)       │                              │   Reward System  │       │
│  └─────────────────┘                              │  & Learning      │       │
│                                                   └──────────────────┘       │
│                                                           │                 │
│                                                           ▼                 │
│                                                  ┌──────────────────┐       │
│                                                  │  Graph Export    │       │
│                                                  │   (Graphviz)     │       │
│                                                  └──────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │        Running Services         │
                    │   (http://localhost:8060/*)     │
                    │                                 │
                    │  ┌─────────┐ ┌─────────────┐    │
                    │  │Employee │ │ Department  │    │
                    │  │Service  │ │  Service    │    │
                    │  └─────────┘ └─────────────┘    │
                    │                                 │
                    │  ┌─────────────────────────┐    │
                    │  │  Organization Service   │    │
                    │  └─────────────────────────┘    │
                    └─────────────────────────────────┘
```

## Data Flow

1. **Input Processing**: CLI accepts OpenAPI spec URLs/paths
2. **Spec Validation**: Parse and validate OpenAPI specifications
3. **Dependency Analysis**: Build hypothesis graph of inter-service dependencies
4. **LLM Enhancement**: Use transformer models to improve hypothesis generation
5. **RL Training**: PPO agent learns optimal test sequences through API interactions
6. **Test Execution**: Execute API calls to verify dependencies and discover bugs
7. **Output Generation**: Export Postman collections and dependency graphs

## Key Components

### RL Environment (gymnasium)
- **State**: Current dependency graph, explored sequences, API responses
- **Actions**: Next endpoint call, sequence variations, hypothesis exploration
- **Rewards**: +10 verified deps, +20 bug discovery, -5 failures, -1 inefficiencies

### Dependency Hypothesis Graph (NetworkX)
- **Nodes**: Endpoints (service:method:path)
- **Edges**: Hypothesized dependencies with confidence scores
- **Analysis**: RESTler-inspired grammar-based data flow detection

### LLM Integration
- **Purpose**: Enhanced hypothesis generation from endpoint descriptions
- **Model**: Transformer-based language model
- **Prompts**: Analyze OpenAPI specs for potential dependencies

## Reward System

| Event | Reward | Description |
|-------|--------|-------------|
| Verified Dependency | +10 | Successful API call confirming hypothesized dependency |
| Bug Discovery | +20 | Uncovered stateful bug (race conditions, invalid sequences) |
| Expected Failure | +5 | 4xx error confirming missing dependency |
| Unexpected Failure | -5 | 5xx error or crash |
| Inefficient Action | -1 | Redundant or non-productive API call | 
"""
Reinforcement Learning Agent for API Testing

This module implements a PPO-based RL agent that learns to generate optimal
API test sequences by interacting with running services. The agent explores
dependency hypotheses, verifies them through API calls, and discovers stateful bugs.
"""

import json
import time
import logging
import numpy as np
import requests
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

try:
    from .spec_parser import ServiceSpec, EndpointInfo
    from .dependency_analyzer import DependencyAnalyzer, DependencyHypothesis
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import ServiceSpec, EndpointInfo
    from dependency_analyzer import DependencyAnalyzer, DependencyHypothesis


logger = logging.getLogger(__name__)


@dataclass
class APICall:
    """Represents an API call and its result."""
    endpoint_id: str
    method: str
    url: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    body: Optional[Dict[str, Any]]
    response_status: Optional[int]
    response_headers: Optional[Dict[str, str]]
    response_body: Optional[Dict[str, Any]]
    response_time: float
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class TestSequence:
    """Represents a sequence of API calls."""
    calls: List[APICall]
    total_reward: float
    verified_dependencies: List[str]
    discovered_bugs: List[Dict[str, Any]]
    sequence_id: str


class APITestEnvironment(gym.Env):
    """Gymnasium environment for API testing with RL."""
    
    def __init__(self, service_specs: List[ServiceSpec], dependency_analyzer: DependencyAnalyzer,
                 base_url: str = "http://localhost:8060", max_sequence_length: int = 20):
        super().__init__()
        
        self.service_specs = service_specs
        self.dependency_analyzer = dependency_analyzer
        self.base_url = base_url
        self.max_sequence_length = max_sequence_length
        
        # Build endpoint mapping
        self.endpoints = {}
        self.endpoint_ids = []
        for spec in service_specs:
            for endpoint in spec.endpoints:
                self.endpoints[endpoint.endpoint_id] = endpoint
                self.endpoint_ids.append(endpoint.endpoint_id)
        
        # Get dependency hypotheses
        self.hypotheses = dependency_analyzer.get_hypotheses()
        self.dependency_graph = dependency_analyzer.graph
        
        # Action space: choose next endpoint to call + optional parameters
        self.action_space = spaces.Discrete(len(self.endpoint_ids) + 1)  # +1 for "done" action
        
        # Observation space: current state representation
        obs_dim = (
            len(self.endpoint_ids) +  # endpoints called
            len(self.hypotheses) +    # hypothesis verification status
            10 +                      # last response features
            5                         # sequence features
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # Session management
        self.session = requests.Session()
        self.session.timeout = 30
        
        # State tracking
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_sequence = []
        self.called_endpoints = set()
        self.verified_hypotheses = set()
        self.discovered_bugs = []
        self.data_store = {}  # Store data from API responses
        self.sequence_reward = 0.0
        self.step_count = 0
        
        # Reset hypothesis verification status
        self.hypothesis_status = {h.producer_endpoint + "->" + h.consumer_endpoint: 0 
                                for h in self.hypotheses}  # -1: failed, 0: unverified, 1: verified
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """Execute an action in the environment."""
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}
        
        # Handle "done" action
        if action >= len(self.endpoint_ids):
            done = True
            reward = self._calculate_sequence_completion_reward()
            return self._get_observation(), reward, done, False, info
        
        # Get selected endpoint
        endpoint_id = self.endpoint_ids[action]
        endpoint = self.endpoints[endpoint_id]
        
        # Check if endpoint was already called (discourage repetition)
        if endpoint_id in self.called_endpoints:
            reward = -1.0  # Small penalty for repetition
            info['action_type'] = 'repetition'
        else:
            # Execute API call
            api_call = self._execute_api_call(endpoint)
            self.current_sequence.append(api_call)
            self.called_endpoints.add(endpoint_id)
            
            # Calculate reward based on call result
            reward = self._calculate_reward(api_call, endpoint)
            info = self._analyze_api_call(api_call, endpoint)
        
        self.sequence_reward += reward
        
        # Check termination conditions
        if (self.step_count >= self.max_sequence_length or 
            len(self.current_sequence) >= self.max_sequence_length):
            done = True
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_api_call(self, endpoint: EndpointInfo) -> APICall:
        """Execute an API call to the specified endpoint."""
        start_time = time.time()
        
        # Build URL
        url = self._build_url(endpoint)
        
        # Prepare headers
        headers = self._prepare_headers(endpoint)
        
        # Prepare parameters
        params = self._prepare_parameters(endpoint)
        
        # Prepare request body
        body = self._prepare_request_body(endpoint)
        
        # Execute request
        try:
            response = self.session.request(
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params.get('query', {}),
                json=body,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            # Parse response
            response_body = None
            try:
                response_body = response.json() if response.content else None
            except:
                response_body = {'raw_content': response.text[:1000]}  # Truncate long responses
            
            api_call = APICall(
                endpoint_id=endpoint.endpoint_id,
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params,
                body=body,
                response_status=response.status_code,
                response_headers=dict(response.headers),
                response_body=response_body,
                response_time=response_time,
                timestamp=start_time,
                success=200 <= response.status_code < 300,
                error_message=None
            )
            
            # Store response data for future use
            if api_call.success and response_body:
                self._store_response_data(endpoint, response_body)
            
            return api_call
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.warning(f"API call failed: {e}")
            
            return APICall(
                endpoint_id=endpoint.endpoint_id,
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params,
                body=body,
                response_status=None,
                response_headers=None,
                response_body=None,
                response_time=response_time,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )
    
    def _build_url(self, endpoint: EndpointInfo) -> str:
        """Build URL for the endpoint, substituting path parameters."""
        # Start with base URL or endpoint's service base URL
        service_spec = next(spec for spec in self.service_specs if spec.service_name == endpoint.service_name)
        base = service_spec.base_url or self.base_url
        
        # Build path with parameter substitution
        path = endpoint.path
        
        # Substitute path parameters with stored data or generated values
        for param in endpoint.parameters:
            if param.get('in') == 'path':
                param_name = param.get('name')
                param_value = self._get_parameter_value(param_name, param.get('schema', {}))
                path = path.replace(f"{{{param_name}}}", str(param_value))
        
        return f"{base.rstrip('/')}{path}"
    
    def _prepare_headers(self, endpoint: EndpointInfo) -> Dict[str, str]:
        """Prepare headers for the API call."""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        
        # Add parameter-based headers
        for param in endpoint.parameters:
            if param.get('in') == 'header':
                param_name = param.get('name')
                param_value = self._get_parameter_value(param_name, param.get('schema', {}))
                headers[param_name] = str(param_value)
        
        # Add authentication headers if available
        if endpoint.security:
            auth_header = self._get_auth_header()
            if auth_header:
                headers.update(auth_header)
        
        return headers
    
    def _prepare_parameters(self, endpoint: EndpointInfo) -> Dict[str, Dict[str, Any]]:
        """Prepare query and path parameters."""
        params = {'query': {}, 'path': {}}
        
        for param in endpoint.parameters:
            param_name = param.get('name')
            param_location = param.get('in')
            param_value = self._get_parameter_value(param_name, param.get('schema', {}))
            
            if param_location == 'query':
                params['query'][param_name] = param_value
            elif param_location == 'path':
                params['path'][param_name] = param_value
        
        return params
    
    def _prepare_request_body(self, endpoint: EndpointInfo) -> Optional[Dict[str, Any]]:
        """Prepare request body for the API call."""
        if not endpoint.request_body:
            return None
        
        # Try to use examples first
        content = endpoint.request_body.get('content', {})
        for media_type, media_info in content.items():
            if 'application/json' in media_type:
                # Use example if available
                if 'example' in media_info:
                    return media_info['example']
                
                # Generate from schema
                schema = media_info.get('schema', {})
                return self._generate_request_body_from_schema(schema)
        
        return None
    
    def _get_parameter_value(self, param_name: str, schema: Dict[str, Any]) -> Any:
        """Get value for a parameter, using stored data or generating new value."""
        # Try to get from stored data first
        if param_name in self.data_store:
            return self.data_store[param_name]
        
        # Try common ID patterns from stored data
        for stored_key, stored_value in self.data_store.items():
            if param_name.lower() in stored_key.lower() or stored_key.lower() in param_name.lower():
                return stored_value
        
        # Generate based on schema
        param_type = schema.get('type', 'string')
        
        if param_type == 'integer':
            return np.random.randint(1, 1000)
        elif param_type == 'number':
            return np.random.uniform(1.0, 1000.0)
        elif param_type == 'boolean':
            return np.random.choice([True, False])
        elif param_type == 'string':
            # Check for ID patterns
            if any(pattern in param_name.lower() for pattern in ['id', 'uuid', 'key']):
                return str(np.random.randint(1, 10000))
            return f"test_{param_name}_{np.random.randint(1, 100)}"
        else:
            return f"test_value_{np.random.randint(1, 100)}"
    
    def _generate_request_body_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate request body from OpenAPI schema."""
        if schema.get('type') == 'object':
            body = {}
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            for prop_name, prop_schema in properties.items():
                # Always include required fields, sometimes include optional ones
                if prop_name in required or np.random.random() < 0.7:
                    body[prop_name] = self._generate_value_from_schema(prop_schema)
            
            return body
        
        return self._generate_value_from_schema(schema)
    
    def _generate_value_from_schema(self, schema: Dict[str, Any]) -> Any:
        """Generate a value based on schema type."""
        schema_type = schema.get('type', 'string')
        
        if schema_type == 'integer':
            minimum = schema.get('minimum', 1)
            maximum = schema.get('maximum', 1000)
            return np.random.randint(minimum, maximum + 1)
        elif schema_type == 'number':
            minimum = schema.get('minimum', 1.0)
            maximum = schema.get('maximum', 1000.0)
            return np.random.uniform(minimum, maximum)
        elif schema_type == 'boolean':
            return np.random.choice([True, False])
        elif schema_type == 'string':
            enum_values = schema.get('enum')
            if enum_values:
                return np.random.choice(enum_values)
            
            format_type = schema.get('format')
            if format_type == 'email':
                return f"test{np.random.randint(1, 1000)}@example.com"
            elif format_type == 'date':
                return "2023-01-01"
            elif format_type == 'date-time':
                return "2023-01-01T12:00:00Z"
            else:
                return f"test_string_{np.random.randint(1, 1000)}"
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            length = np.random.randint(1, 4)  # Generate 1-3 items
            return [self._generate_value_from_schema(items_schema) for _ in range(length)]
        elif schema_type == 'object':
            return self._generate_request_body_from_schema(schema)
        else:
            return f"unknown_type_{np.random.randint(1, 1000)}"
    
    def _store_response_data(self, endpoint: EndpointInfo, response_body: Dict[str, Any]):
        """Store useful data from API response for future use."""
        if not isinstance(response_body, dict):
            return
        
        # Store ID-like fields
        for key, value in response_body.items():
            if any(pattern in key.lower() for pattern in ['id', 'uuid', 'key', 'ref']):
                self.data_store[key] = value
        
        # Store arrays of objects
        for key, value in response_body.items():
            if isinstance(value, list) and value:
                # Store first item's ID fields
                first_item = value[0]
                if isinstance(first_item, dict):
                    for item_key, item_value in first_item.items():
                        if any(pattern in item_key.lower() for pattern in ['id', 'uuid', 'key']):
                            self.data_store[f"{key}_{item_key}"] = item_value
    
    def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get authentication header if token is available."""
        # Try to get token from stored data
        for key, value in self.data_store.items():
            if 'token' in key.lower():
                return {'Authorization': f'Bearer {value}'}
        
        # Return empty for now - in real scenarios, this would handle auth flows
        return None
    
    def _calculate_reward(self, api_call: APICall, endpoint: EndpointInfo) -> float:
        """Calculate reward for an API call."""
        reward = 0.0
        
        # Base reward for successful calls
        if api_call.success:
            reward += 5.0
            
            # Bonus for data production (POST, PUT with response data)
            if endpoint.method in ['POST', 'PUT'] and api_call.response_body:
                reward += 5.0
        else:
            # Handle different types of failures
            if api_call.response_status:
                if 400 <= api_call.response_status < 500:
                    # Client errors - might indicate dependency issues
                    reward += 2.0  # Small positive reward for expected failures
                elif 500 <= api_call.response_status < 600:
                    # Server errors - potential bugs
                    reward += 15.0  # High reward for discovering server issues
                    self._record_potential_bug(api_call, "server_error")
                else:
                    reward -= 2.0
            else:
                # Network/connection errors
                reward -= 5.0
        
        # Reward for verifying dependencies
        dependency_reward = self._check_dependency_verification(api_call, endpoint)
        reward += dependency_reward
        
        # Penalty for inefficient actions
        if len(self.current_sequence) > 1:
            # Check if this call seems redundant
            if self._is_redundant_call(api_call, endpoint):
                reward -= 1.0
        
        return reward
    
    def _check_dependency_verification(self, api_call: APICall, endpoint: EndpointInfo) -> float:
        """Check if this API call verifies any dependency hypotheses."""
        reward = 0.0
        
        # Check if this call acts as a consumer in any hypothesis
        for hypothesis in self.hypotheses:
            if hypothesis.consumer_endpoint == endpoint.endpoint_id:
                hypothesis_key = f"{hypothesis.producer_endpoint}->{hypothesis.consumer_endpoint}"
                
                # Check if producer was called before this consumer
                producer_called = any(call.endpoint_id == hypothesis.producer_endpoint 
                                    for call in self.current_sequence[:-1])  # Exclude current call
                
                if producer_called and hypothesis_key not in self.verified_hypotheses:
                    if api_call.success:
                        # Successful call after producer suggests dependency is correct
                        reward += 10.0 * hypothesis.confidence
                        self.verified_hypotheses.add(hypothesis_key)
                        self.hypothesis_status[hypothesis_key] = 1
                    else:
                        # Failed call after producer might indicate dependency issue
                        if api_call.response_status and 400 <= api_call.response_status < 500:
                            reward += 5.0 * hypothesis.confidence
                            self.hypothesis_status[hypothesis_key] = -1
        
        return reward
    
    def _record_potential_bug(self, api_call: APICall, bug_type: str):
        """Record a potential bug discovery."""
        bug = {
            'type': bug_type,
            'endpoint': api_call.endpoint_id,
            'status_code': api_call.response_status,
            'sequence_position': len(self.current_sequence),
            'sequence_context': [call.endpoint_id for call in self.current_sequence[-3:]],  # Last 3 calls
            'timestamp': api_call.timestamp,
            'response_body': api_call.response_body
        }
        self.discovered_bugs.append(bug)
    
    def _is_redundant_call(self, api_call: APICall, endpoint: EndpointInfo) -> bool:
        """Check if the API call is redundant."""
        # Simple heuristic: same endpoint called recently with similar parameters
        recent_calls = self.current_sequence[-5:]  # Check last 5 calls
        
        for call in recent_calls[:-1]:  # Exclude current call
            if (call.endpoint_id == api_call.endpoint_id and 
                call.method == api_call.method):
                return True
        
        return False
    
    def _calculate_sequence_completion_reward(self) -> float:
        """Calculate reward for completing a sequence."""
        reward = 0.0
        
        # Reward for verified dependencies
        reward += len(self.verified_hypotheses) * 5.0
        
        # Reward for discovered bugs
        reward += len(self.discovered_bugs) * 10.0
        
        # Penalty for very short sequences (encourage exploration)
        if len(self.current_sequence) < 3:
            reward -= 5.0
        
        return reward
    
    def _analyze_api_call(self, api_call: APICall, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Analyze API call results for additional insights."""
        info = {
            'action_type': 'api_call',
            'endpoint_id': api_call.endpoint_id,
            'success': api_call.success,
            'status_code': api_call.response_status,
            'response_time': api_call.response_time
        }
        
        # Check for potential race conditions or state issues
        if not api_call.success and api_call.response_status == 409:  # Conflict
            info['potential_race_condition'] = True
            self._record_potential_bug(api_call, "race_condition")
        
        # Check for timeout issues
        if api_call.response_time > 5.0:  # Slow response
            info['slow_response'] = True
        
        return info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        obs = []
        
        # Endpoints called (binary vector)
        for endpoint_id in self.endpoint_ids:
            obs.append(1.0 if endpoint_id in self.called_endpoints else 0.0)
        
        # Hypothesis verification status
        for hypothesis_key in sorted(self.hypothesis_status.keys()):
            obs.append(float(self.hypothesis_status[hypothesis_key]))
        
        # Last response features
        if self.current_sequence:
            last_call = self.current_sequence[-1]
            obs.extend([
                1.0 if last_call.success else -1.0,
                (last_call.response_status or 0) / 500.0,  # Normalize status code
                min(last_call.response_time / 5.0, 1.0),   # Normalize response time
                1.0 if last_call.response_body else 0.0,
                len(self.data_store) / 100.0  # Normalize data store size
            ])
        else:
            obs.extend([0.0] * 5)  # No calls yet
        
        # Add padding to reach exactly 10 features
        obs.extend([0.0] * (10 - 5))
        
        # Sequence features
        obs.extend([
            len(self.current_sequence) / self.max_sequence_length,
            len(self.verified_hypotheses) / max(len(self.hypotheses), 1),
            len(self.discovered_bugs) / 10.0,  # Normalize bug count
            self.sequence_reward / 100.0,  # Normalize reward
            self.step_count / self.max_sequence_length
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def get_current_sequence(self) -> TestSequence:
        """Get current test sequence."""
        return TestSequence(
            calls=self.current_sequence.copy(),
            total_reward=self.sequence_reward,
            verified_dependencies=list(self.verified_hypotheses),
            discovered_bugs=self.discovered_bugs.copy(),
            sequence_id=f"seq_{int(time.time())}"
        )


class RLAgent:
    """PPO-based RL agent for API testing."""
    
    def __init__(self, service_specs: List[ServiceSpec], dependency_analyzer: DependencyAnalyzer,
                 base_url: str = "http://localhost:8060"):
        self.service_specs = service_specs
        self.dependency_analyzer = dependency_analyzer
        self.base_url = base_url
        
        # Create environment
        self.env = APITestEnvironment(service_specs, dependency_analyzer, base_url)
        
        # Check if tensorboard is available
        tensorboard_log = None
        try:
            import tensorboard
            tensorboard_log = "./tensorboard_logs/"
            logger.info("TensorBoard logging enabled")
        except ImportError:
            logger.warning("TensorBoard not available, logging disabled")
        
        # Create PPO agent
        self.agent = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tensorboard_log
        )
        
        # Training history
        self.training_history = []
        self.generated_sequences = []
    
    def train(self, total_timesteps: int = 50000) -> None:
        """Train the RL agent."""
        logger.info(f"Starting RL training for {total_timesteps} timesteps...")
        
        # Custom callback to track progress
        callback = TrainingCallback(self)
        
        # Check if progress bar dependencies are available
        progress_bar = True
        try:
            import rich
            logger.info("Progress bar enabled")
        except ImportError:
            logger.warning("Rich not available, disabling progress bar")
            progress_bar = False
        
        # Train the agent
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar
        )
        
        logger.info("Training completed!")
    
    def generate_test_sequence(self, max_length: int = 20) -> TestSequence:
        """Generate a test sequence using the trained agent."""
        obs, _ = self.env.reset()
        sequence = []
        
        for step in range(max_length):
            action, _ = self.agent.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = self.env.step(action)
            
            if done:
                break
        
        test_sequence = self.env.get_current_sequence()
        self.generated_sequences.append(test_sequence)
        return test_sequence
    
    def generate_multiple_sequences(self, num_sequences: int = 10) -> List[TestSequence]:
        """Generate multiple test sequences."""
        sequences = []
        for i in range(num_sequences):
            logger.info(f"Generating sequence {i+1}/{num_sequences}")
            sequence = self.generate_test_sequence()
            sequences.append(sequence)
        
        return sequences
    
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        self.agent.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.agent = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")


class TrainingCallback(BaseCallback):
    """Custom callback for training progress tracking."""
    
    def __init__(self, rl_agent: RLAgent):
        super().__init__()
        self.rl_agent = rl_agent
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Track episode statistics
        if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
            episode_reward = sum(self.locals.get('rewards', []))
            self.episode_rewards.append(episode_reward)
            
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Episode {len(self.episode_rewards)}: Average reward = {avg_reward:.2f}")
        
        return True


def main():
    """Example usage of the RL agent."""
    import sys
    try:
        from .spec_parser import SpecParser
        from .dependency_analyzer import DependencyAnalyzer
    except ImportError:
        from spec_parser import SpecParser
        from dependency_analyzer import DependencyAnalyzer
    
    if len(sys.argv) < 2:
        print("Usage: python rl_agent.py <spec_url_or_path> [<spec_url_or_path> ...]")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse specs
    parser = SpecParser()
    specs = parser.parse_specs(sys.argv[1:])
    
    if not specs:
        print("No valid specifications found")
        sys.exit(1)
    
    # Analyze dependencies
    analyzer = DependencyAnalyzer()
    analyzer.analyze_dependencies(specs)
    
    # Create and train RL agent
    agent = RLAgent(specs, analyzer)
    
    print("Training RL agent...")
    agent.train(total_timesteps=10000)  # Reduced for demo
    
    print("\nGenerating test sequences...")
    sequences = agent.generate_multiple_sequences(5)
    
    print(f"\nGenerated {len(sequences)} test sequences:")
    for i, seq in enumerate(sequences, 1):
        print(f"\nSequence {i}:")
        print(f"  Calls: {len(seq.calls)}")
        print(f"  Total reward: {seq.total_reward:.2f}")
        print(f"  Verified dependencies: {len(seq.verified_dependencies)}")
        print(f"  Discovered bugs: {len(seq.discovered_bugs)}")
        
        print("  Call sequence:")
        for call in seq.calls:
            status = "✓" if call.success else "✗"
            print(f"    {status} {call.method} {call.endpoint_id} -> {call.response_status}")


if __name__ == "__main__":
    main() 
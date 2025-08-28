"""
Enhanced version of your llm_action_planner.py
Only adds RAG functionality to the existing LLMActionPlanner class
Minimal changes to your existing code
"""

import sys
from os import path
import yaml
import logging
import json
from dotenv import dotenv_values
from openai import OpenAI
from tenacity import retry, stop_after_attempt
import jinja2
import difflib
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import re
from collections import Counter
import validate_responses

# Add parent directories dynamically
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
)
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from AIDojoCoordinator.game_components import ActionType, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state


class ConfigLoader:
    """Class to handle loading YAML configurations."""

    @staticmethod
    def load_config(file_name: str = 'prompts.yaml') -> dict:
        possible_paths = [
            path.join(path.dirname(__file__), file_name),
            path.join(path.dirname(path.dirname(__file__)), file_name),
            path.join(path.dirname(path.dirname(path.dirname(__file__))), file_name),
        ]
        for yaml_file in possible_paths:
            if path.exists(yaml_file):
                with open(yaml_file, 'r') as file:
                    return yaml.safe_load(file)
        raise FileNotFoundError(f"{file_name} not found in expected directories.")


ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}


class NetSecGameRAG:
    """
    RAG system for NetSecGame episodes
    Handles episodes in format: [state, prompt, response, evaluation]
    Can load from single file or directory of JSON files

    This class implements pattern-based matching of cyber attack states across different game executions.
    The key method normalize_cyber_state transforms concrete game states (with specific IPs, hostnames, etc.)
    into abstract patterns representing asset types, service profiles, attack phases, and network topology ratios,
    allowing robust matching despite instance-specific differences.
    """
    def __init__(self, episodes_path: str = None, similarity_threshold: float = 0.70, good_action_threshold: int = 8):
        self.similarity_threshold = similarity_threshold
        self.good_action_threshold = good_action_threshold
        self.past_episodes = []
        self.good_actions_by_state = defaultdict(list)
        self.good_actions_by_goal_and_state = defaultdict(lambda: defaultdict(list))
        self.logger = logging.getLogger("NetSecGameRAG")

        if episodes_path:
            self.load_episodes(episodes_path)
            self.index_good_actions()

    def load_episodes(self, episodes_path: str):
        """Load episodes from single file or directory of JSON files"""
        import glob
        import os

        try:
            if os.path.isfile(episodes_path):
                with open(episodes_path, 'r') as f:
                    episodes = json.load(f)

                # If single file is dict with lists, convert to list of episode lists
                if isinstance(episodes, dict) and all(key in episodes for key in ['state', 'prompt', 'response', 'evaluation']):
                    num_records = len(episodes['state'])
                    if not (len(episodes['prompt']) == num_records and
                            len(episodes['response']) == num_records and
                            len(episodes['evaluation']) == num_records):
                        self.logger.warning(f"Skipping {episodes_path} - lists have inconsistent lengths.")
                        self.past_episodes = []
                        return

                    self.past_episodes = [[
                        episodes['state'][i],
                        episodes['prompt'][i],
                        episodes['response'][i],
                        episodes['evaluation'][i]
                    ] for i in range(num_records)]
                    self.logger.info(f"RAG loaded {num_records} episodes from {episodes_path}")

                elif isinstance(episodes, list):
                    self.past_episodes = episodes
                    self.logger.info(f"RAG loaded {len(self.past_episodes)} episodes from {episodes_path}")

                else:
                    self.logger.warning(f"Skipping {episodes_path} - unexpected JSON structure")
                    self.past_episodes = []

            elif os.path.isdir(episodes_path):
                json_files = glob.glob(os.path.join(episodes_path, "*.json"))
                total_episodes = 0

                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            episodes = json.load(f)
                            if isinstance(episodes, dict) and all(key in episodes for key in ['state', 'prompt', 'response', 'evaluation']):
                                num_records = len(episodes['state'])
                                if not (len(episodes['prompt']) == num_records and
                                        len(episodes['response']) == num_records and
                                        len(episodes['evaluation']) == num_records):
                                    self.logger.warning(f"Skipping {json_file} - lists have inconsistent lengths.")
                                    continue

                                for i in range(num_records):
                                    self.past_episodes.append([
                                        episodes['state'][i],
                                        episodes['prompt'][i],
                                        episodes['response'][i],
                                        episodes['evaluation'][i]
                                    ])
                                total_episodes += num_records
                                self.logger.debug(f"Loaded {num_records} records from {json_file}")
                            elif isinstance(episodes, list):
                                self.past_episodes.extend(episodes)
                                total_episodes += len(episodes)
                                self.logger.debug(f"Loaded {len(episodes)} episodes from {json_file}")
                            else:
                                self.logger.warning(f"Skipping {json_file} - unexpected JSON structure")
                    except Exception as e:
                        self.logger.warning(f"Could not load {json_file}: {e}")

                self.logger.info(f"RAG loaded {total_episodes} total episodes from {len(json_files)} files in {episodes_path}")
            else:
                self.logger.warning(f"Episodes path {episodes_path} does not exist")

        except Exception as e:
            self.logger.warning(f"Could not load RAG episodes from {episodes_path}: {e}")
            self.past_episodes = []


    def index_good_actions(self):
        """Pre-index good actions by state and goal for faster lookup"""
        self.good_actions_by_state.clear()
        self.good_actions_by_goal_and_state.clear()

        if isinstance(self.past_episodes, dict):
            # Handle dictionary structure with lists
            states = self.past_episodes.get('state', [])
            responses = self.past_episodes.get('response', [])
            evaluations = self.past_episodes.get('evaluation', [])
            goals = self.past_episodes.get('goal', [None] * len(states))  # default None if goal missing

            # Validate all lists have the same length
            if not (len(states) == len(responses) == len(evaluations) == len(goals)):
                self.logger.warning("Inconsistent lengths in past_episodes dictionary lists including goal.")
                return

            for state, response, evaluation, goal in zip(states, responses, evaluations, goals):
                if evaluation >= self.good_action_threshold:
                    state_str = json.dumps(state, sort_keys=True)
                    try:
                        response_dict = json.loads(response) if isinstance(response, str) else response
                        action_info = {
                            'action': response_dict.get('action', 'Unknown'),
                            'parameters': response_dict.get('parameters', {}),
                            'evaluation': evaluation
                        }
                        self.good_actions_by_state[state_str].append(action_info)
                        if goal is not None:
                            self.good_actions_by_goal_and_state[goal][state_str].append(action_info)
                    except (json.JSONDecodeError, TypeError):
                        self.logger.debug(f"Failed to parse response or action from episode record with state {state}")
                        continue
        else:
            # Assume list of episode records
            for episode_record in self.past_episodes:
                # Handle dict type episode record
                if isinstance(episode_record, dict):
                    state = episode_record.get('state')
                    response = episode_record.get('response')
                    evaluation = episode_record.get('evaluation')
                    goal = episode_record.get('goal', None)

                    # If evaluation is list, flatten and index individually
                    if isinstance(evaluation, list):
                        # Ensure state and response are lists of the same length
                        states = state if isinstance(state, list) else [state]
                        responses = response if isinstance(response, list) else [response]
                        evaluations = evaluation
                        goals = [goal] * len(evaluations) if goal is not None else [None] * len(evaluations)

                        if not (len(states) == len(responses) == len(evaluations)):
                            self.logger.warning("Inconsistent lengths when flattening episode record with lists.")
                            continue

                        for s, r, e, g in zip(states, responses, evaluations, goals):
                            if e is not None and e >= self.good_action_threshold:
                                state_str = json.dumps(s, sort_keys=True)
                                try:
                                    response_dict = json.loads(r) if isinstance(r, str) else r
                                    action_info = {
                                        'action': response_dict.get('action', 'Unknown'),
                                        'parameters': response_dict.get('parameters', {}),
                                        'evaluation': e
                                    }
                                    self.good_actions_by_state[state_str].append(action_info)
                                    if g is not None:
                                        self.good_actions_by_goal_and_state[g][state_str].append(action_info)
                                except (json.JSONDecodeError, TypeError):
                                    self.logger.debug(f"Failed to parse response or action from episode record (flattened part): {episode_record}")
                                    continue
                        continue

                # Otherwise fallback to list/tuple type
                if len(episode_record) >= 4:
                    state = episode_record[0]
                    response = episode_record[2]
                    evaluation = episode_record[3]
                    goal = None
                    if isinstance(state, dict) and 'goal' in state:
                        goal = state['goal']

                    if evaluation is not None and evaluation >= self.good_action_threshold:
                        state_str = json.dumps(state, sort_keys=True)
                        try:
                            response_dict = json.loads(response) if isinstance(response, str) else response
                            action_info = {
                                'action': response_dict.get('action', 'Unknown'),
                                'parameters': response_dict.get('parameters', {}),
                                'evaluation': evaluation
                            }
                            self.good_actions_by_state[state_str].append(action_info)
                            if goal is not None:
                                self.good_actions_by_goal_and_state[goal][state_str].append(action_info)
                        except (json.JSONDecodeError, TypeError):
                            self.logger.debug(f"Failed to parse response or action from episode record: {episode_record}")
                            continue

        # Also index by abstract patterns for better cross-execution matching
        self.good_actions_by_pattern = defaultdict(list)

        for state_str, actions in self.good_actions_by_state.items():
            try:
                state = json.loads(state_str)
                pattern = self.normalize_cyber_state(state)
                pattern_key = json.dumps(pattern, sort_keys=True)

                for action_info in actions:
                    self.good_actions_by_pattern[pattern_key].append(action_info)

            except (json.JSONDecodeError, Exception) as e:
                self.logger.debug(f"Could not create pattern index for state: {state_str} - {e}")
                continue

    def calculate_state_similarity(self, state1: Dict, state2: Dict) -> float:
        """Calculate similarity between two NetSecGame states"""
        # Convert to JSON strings for comparison
        state1_str = json.dumps(state1, sort_keys=True)
        state2_str = json.dumps(state2, sort_keys=True)

        # Basic string similarity
        string_similarity = difflib.SequenceMatcher(None, state1_str, state2_str).ratio()

        # Semantic similarity based on pattern matching
        normalized_pattern1 = self.normalize_cyber_state(state1)
        normalized_pattern2 = self.normalize_cyber_state(state2)
        semantic_similarity = self.calculate_pattern_similarity(normalized_pattern1, normalized_pattern2)

        # Weighted combination
        return 0.3 * string_similarity + 0.7 * semantic_similarity

    def calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two normalized cyber state patterns"""
        def jaccard(set1, set2):
            if not set1 and not set2:
                return 1.0
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0

        weights = {
            'asset_type': 0.3,
            'attack_phase': 0.25,
            'service_profile': 0.2,
            'topology': 0.1,
            'goal': 0.15
        }

        # Asset Type similarity
        atypes1 = set(pattern1.get('asset_types', []))
        atypes2 = set(pattern2.get('asset_types', []))
        asset_type_similarity = jaccard(atypes1, atypes2)

        # Attack Phase similarity
        phase1 = pattern1.get('attack_phase', '')
        phase2 = pattern2.get('attack_phase', '')
        attack_phase_similarity = 1.0 if phase1 == phase2 and phase1 != '' else 0.0

        # Service Profile similarity
        sp1 = pattern1.get('service_profiles', {})
        sp2 = pattern2.get('service_profiles', {})
        common_asset_types = atypes1.intersection(atypes2)

        if common_asset_types:
            service_similarities = []
            for atype in common_asset_types:
                s1 = set(sp1.get(atype, []))
                s2 = set(sp2.get(atype, []))
                service_similarities.append(jaccard(s1, s2))
            service_profile_similarity = sum(service_similarities) / len(service_similarities)
        else:
            service_profile_similarity = 0.0

        # Topology similarity (based on ratios)
        top1 = pattern1.get('topology_ratios', {})
        top2 = pattern2.get('topology_ratios', {})
        topology_scores = []

        for key in top1:
            if key in top2:
                diff = abs(top1[key] - top2.get(key, 0))
                topology_scores.append(1 - diff)  # closer ratio -> higher similarity

        topology_similarity = np.mean(topology_scores) if topology_scores else 0.0

        # Goal similarity
        goal1 = pattern1.get('goal', '')
        goal2 = pattern2.get('goal', '')
        goal_similarity = 1.0 if goal1 == goal2 and goal1 != '' else 0.0

        # Weighted sum
        similarity = (weights['asset_type'] * asset_type_similarity +
                      weights['attack_phase'] * attack_phase_similarity +
                      weights['service_profile'] * service_profile_similarity +
                      weights['topology'] * topology_similarity +
                      weights['goal'] * goal_similarity)

        return similarity

    def _normalize_component(self, component):
        """Normalize state component to comparable format"""
        if isinstance(component, list):
            return set(str(item) for item in component)
        elif isinstance(component, dict):
            return set(f"{k}:{v}" for k, v in component.items())
        else:
            return {str(component)}

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def normalize_cyber_state(self, state: Dict) -> Dict:
        """Normalize concrete cyber game state into an abstract pattern for cross execution matching.

        Extract asset types, service profiles, attack phase, topology ratios, and goal while removing instance specifics.
        """
        # Defensive copy
        normalized = {}

        known_services = state.get('known_services', {})
        controlled_hosts = state.get('controlled_hosts', [])
        known_hosts = state.get('known_hosts', [])

        # Extract IP addresses from known_hosts (which is a list of dicts)
        known_hosts_ips = set()
        for host_info in known_hosts:
            if isinstance(host_info, dict) and 'ip' in host_info:
                known_hosts_ips.add(host_info['ip'])
            elif isinstance(host_info, str):
                known_hosts_ips.add(host_info)

        # Classify controlled hosts by asset type
        asset_types = []
        service_profiles = {}

        for host in controlled_hosts:
            # Extract host IP if it's a dict, otherwise use as is
            host_ip = host['ip'] if isinstance(host, dict) and 'ip' in host else host
            host_services_raw = known_services.get(host_ip, [])
            # Extract generic service names ignoring versions
            host_services = self._get_service_profile_raw(host_services_raw)
            asset_type = self._classify_asset(host_services)
            asset_types.append(asset_type)
            service_profiles.setdefault(asset_type, [])
            for svc in host_services:
                if svc not in service_profiles[asset_type]:
                    service_profiles[asset_type].append(svc)

        # De-duplicate
        asset_types = sorted(set(asset_types))

        # Calculate topology ratios
        known_hosts_count = len(known_hosts_ips) if known_hosts_ips else 1

        # Extract controlled host IPs
        controlled_ips = set()
        for host in controlled_hosts:
            if isinstance(host, dict) and 'ip' in host:
                controlled_ips.add(host['ip'])
            elif isinstance(host, str):
                controlled_ips.add(host)

        controlled_count = len(controlled_ips)
        controlled_to_known_ratio = controlled_count / known_hosts_count

        # Server to workstation ratio (count assets)
        def count_asset_type(asset_type_name):
            return sum(1 for at in asset_types if at == asset_type_name)

        server_count = count_asset_type('database_server') + count_asset_type('web_server') + count_asset_type(
            'domain_controller')
        workstation_count = count_asset_type('workstation') or 1  # avoid div by zero
        server_to_workstation_ratio = server_count / workstation_count

        topology_ratios = {
            'controlled_to_known_ratio': controlled_to_known_ratio,
            'server_to_workstation_ratio': server_to_workstation_ratio
        }

        # Determine attack phase
        attack_phase = self._determine_attack_phase(state)

        # Extract goal
        goal = state.get('goal', '')

        normalized['asset_types'] = asset_types
        normalized['service_profiles'] = {atype: sorted(svcs) for atype, svcs in service_profiles.items()}
        normalized['attack_phase'] = attack_phase
        normalized['topology_ratios'] = topology_ratios
        normalized['goal'] = goal

        return normalized

    def _get_service_profile_raw(self, raw_services: List) -> List[str]:
        """Parse raw known services to generic names removing versions and duplicates"""
        services = set()
        for svc in raw_services:
            # Handle both string and dictionary service formats
            if isinstance(svc, dict) and 'name' in svc:
                # Extract service base name without version numbers from dictionary
                svc_name = svc['name'].lower().split()[0] if svc['name'] else ''
                if svc_name:
                    services.add(svc_name)
            elif isinstance(svc, str):
                # Extract service base name without version numbers from string
                svc_base = svc.lower().split()[0] if svc else ''
                if svc_base:
                    services.add(svc_base)
        return sorted(services)

    def _classify_asset(self, service_list: List[str]) -> str:
        """Classify a single host asset type based on its generic service list"""
        service_set = set(service_list)

        if {'postgresql', 'ssh'}.issubset(service_set):
            return 'database_server'
        if {'http', 'https', 'ssh'}.intersection(service_set):
            return 'web_server'
        if {'kerberos', 'ldap'}.issubset(service_set):
            return 'domain_controller'
        if 'ssh' in service_set and len(service_set) == 1:
            return 'workstation'
        if {'snmp', 'telnet'}.intersection(service_set):
            return 'network_device'
        return 'unknown'

    def _determine_attack_phase(self, state: Dict) -> str:
        """Determine the current attack phase based on controlled hosts and their types"""
        controlled_hosts = state.get('controlled_hosts', [])
        known_services = state.get('known_services', {})

        if not controlled_hosts:
            return 'reconnaissance'

        # Classify controlled hosts
        controlled_types = set()
        for host in controlled_hosts:
            # Extract host IP if it's a dict, otherwise use as is
            host_ip = host['ip'] if isinstance(host, dict) and 'ip' in host else host

            services_raw = known_services.get(host_ip, [])
            services = self._get_service_profile_raw(services_raw)
            asset_type = self._classify_asset(services)
            controlled_types.add(asset_type)

        if controlled_types == {'workstation'}:
            return 'initial_compromise'
        if 'domain_controller' in controlled_types:
            # Assume goal achieved or privilege escalation if domain controller controlled
            return 'goal_achieved'
        if 'workstation' in controlled_types and len(controlled_types) > 1:
            return 'lateral_movement'
        if 'database_server' in controlled_types:
            return 'data_exfiltration_ready'

        return 'privilege_escalation'

    def get_top_k_recommendations(self, current_state: Dict, k: int = 3, goal: str | None = None) -> list[
        tuple[float, dict]]:
        """Retrieve top K recommendations based on similarity, optionally filtered by goal."""
        # Ensure current_state is a dict
        if isinstance(current_state, str):
            try:
                current_state = json.loads(current_state)
            except json.JSONDecodeError:
                self.logger.warning(f"current_state string could not be decoded to dict: {current_state}")
                return []

        all_recommendations = []

        if goal and goal in self.good_actions_by_goal_and_state:
            candidate_states = self.good_actions_by_goal_and_state[goal]
        else:
            candidate_states = self.good_actions_by_state

        for state_str, actions in candidate_states.items():
            try:
                # First decode the state string
                state_dict = json.loads(state_str)

                # If the result is still a string, it was double-encoded
                if isinstance(state_dict, str):
                    try:
                        state_dict = json.loads(state_dict)  # Decode again
                    except json.JSONDecodeError:
                        self.logger.debug(f"Failed to decode double-encoded state string: {state_str}")
                        continue

                if not isinstance(state_dict, dict):
                    self.logger.debug(f"Skipping state during recommendation because it is not dict: {state_str}")
                    continue

                similarity = self.calculate_state_similarity(current_state, state_dict)

                if similarity >= self.similarity_threshold:
                    for action_info in actions:
                        all_recommendations.append((similarity, action_info))
            except json.JSONDecodeError:
                self.logger.debug(f"Failed to decode state string during recommendation: {state_str}")

        all_recommendations.sort(key=lambda x: x[0], reverse=True)
        return all_recommendations[:k]

    def get_action_recommendation(self, current_state: Dict, k: int = 3, goal: str | None = None, recent_memory: list[dict] | None = None) -> tuple[bool, str]:
        """Get action recommendation with top-k, goal-aware filtering, and memory integration."""
        initial_k = 5  # Increase retrieval size to get a wider pool
        recommendations = self.get_top_k_recommendations(current_state, k=initial_k, goal=goal)

        failed_actions = set()
        def extract_mem_dict(mem):
            while isinstance(mem, tuple) and len(mem) > 0:
                mem = mem[0]
            if isinstance(mem, dict):
                return mem
            return {}

        if recent_memory:
            for mem in recent_memory:
                mem_dict = extract_mem_dict(mem)
                if mem_dict.get('helpful') is False or mem_dict.get('valid') is False:
                    failed_actions.add(mem_dict.get('action'))

        filtered_recommendations = [rec for rec in recommendations if rec[1].get('action') not in failed_actions]
        filtered_recommendations = filtered_recommendations[:k]  # limit after filtering

        if filtered_recommendations:
            formatted_recommendations = []
            for similarity, action_info in filtered_recommendations:
                formatted_recommendations.append(
                    f"  - Action: {action_info['action']}, Parameters: {action_info['parameters']}"
                )
            recommendation_text = "Based on similar past states, highly rated actions were:\n" + "\n".join(formatted_recommendations)
            return True, recommendation_text

        return False, ""


class LLMActionPlanner:
    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None, config: dict = None,
                 use_reasoning: bool = False, use_reflection: bool = False, use_self_consistency: bool = False,
                 episodes_json_file: str = None, rag_similarity_threshold: float = 0.72, rag_good_threshold: int = 8):
        """
        Enhanced LLM Action Planner with RAG capabilities

        New parameters:
            episodes_json_file: Path to episodes JSON file OR directory containing multiple JSON files
            rag_similarity_threshold: Minimum similarity to consider states similar (0.85 recommended)
            rag_good_threshold: Minimum evaluation score to consider action "good" (8 recommended)
        """
        self.model = model_name
        self.config = config or ConfigLoader.load_config()
        self.use_reasoning = use_reasoning
        self.use_reflection = use_reflection
        self.use_self_consistency = use_self_consistency

        if "gpt" in self.model:
            env_config = dotenv_values(".env")
            self.client = OpenAI(api_key=env_config["OPENAI_API_KEY"])
        else:
            self.client = OpenAI(base_url=api_url, api_key="ollama")

        self.memory_len = memory_len
        self.logger = logging.getLogger("REACT-agent")
        self.update_instructions(goal.lower())
        self.prompts = []
        self.states = []
        self.responses = []

        # Initialize RAG system
        if episodes_json_file:
            self.rag_system = NetSecGameRAG(
                episodes_path=episodes_json_file,  # Changed parameter name
                similarity_threshold=rag_similarity_threshold,
                good_action_threshold=rag_good_threshold
            )
            if self.rag_system.past_episodes:
                self.logger.info(f"RAG system initialized with {len(self.rag_system.past_episodes)} episodes")
            else:
                self.logger.warning("RAG system initialized but no episodes loaded")
        else:
            self.rag_system = None

    def get_prompts(self) -> list:
        """Returns the list of prompts sent to the LLM."""
        return self.prompts

    def get_responses(self) -> list:
        """Returns the list of responses received from the LLM. Only Stage 2 responses are included."""
        return self.responses

    def get_states(self) -> list:
        """Returns the list of states received from the LLM. In JSON format."""
        return self.states

    def update_instructions(self, new_goal: str) -> None:
        template = jinja2.Environment().from_string(self.config['prompts']['INSTRUCTIONS_TEMPLATE'])
        self.instructions = template.render(goal=new_goal)

    def create_mem_prompt(self, memory_list: list) -> str:
        prompt = ""
        for memory, goodness in memory_list:
            prompt += f"You have taken action {memory} in the past. This action was {goodness}.\n"
        return prompt

    @retry(stop=stop_after_attempt(3))
    def openai_query(self, msg_list: list, max_tokens: int = 60, model: str = None, fmt=None, temperature: float = 0.0):
        llm_response = self.client.chat.completions.create(
            model=model or self.model,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=fmt or {"type": "text"},
        )
        return llm_response.choices[0].message.content

    def parse_response_deprecated(self, llm_response: str, state: Observation.state):
        try:
            response = json.loads(llm_response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON.")
            return False,llm_response, None

        try:
            action_str = response["action"]
            action_params = response["parameters"]
            valid, action = create_action_from_response(response, state)
            return valid, {action_str:action_str,action_params:action_params}, action

        except KeyError:
            return False, llm_response, None

    def parse_response(self, llm_response: str, state: Observation.state):
        response_dict = {"action": None, "parameters": None}
        valid = False
        action = None

        try:
            response = json.loads(llm_response)
            action_str = response.get("action", None)
            action_params = response.get("parameters", None)

            if action_str and action_params:
                valid, action = create_action_from_response(response, state)
                response_dict["action"] = action_str
                response_dict["parameters"] = action_params
            else:
                self.logger.warning("Missing action or parameters in LLM response.")
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON.")
            response_dict["action"] = "InvalidJSON"
            response_dict["parameters"] = llm_response
        except KeyError:
            self.logger.error("Missing keys in LLM response.")

        return valid, response_dict, action

    def remove_reasoning(self, text):
        match = re.search(r'</think>(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def check_repetition(self, memory_list):
        repetitions = 0
        past_memories = []
        for memory, goodness in memory_list:
            if memory in past_memories:
                repetitions += 1
            past_memories.append(memory)
        return repetitions

    def get_self_consistent_response(self, messages, temp=0.4, max_tokens=1024, n=3):
        candidates = []
        for _ in range(n):
            response = self.openai_query(messages, temperature=temp, max_tokens=max_tokens)
            candidates.append(response.strip())

        counts = Counter(candidates)
        most_common = counts.most_common(1)
        if most_common:
            self.logger.info(f"Self-consistency candidates: {counts}")
            return most_common[0][0]
        return candidates[0]

    def check_previous_states(self, current_state_str: str) -> tuple[bool, str]:
        """
        Enhanced version using RAG system
        Check if a similar state exists in past good actions and return recommendation
        """
        # Try RAG system first
        if self.rag_system:
            try:
                current_state = json.loads(current_state_str)
                found, recommendation = self.rag_system.get_action_recommendation(current_state)

                if found:
                    self.logger.info(f"RAG found similar state with recommendation")
                    return found, recommendation
                else:
                    self.logger.info("RAG: No similar states found")

            except json.JSONDecodeError:
                self.logger.error("Could not parse current state for RAG lookup")

        # Fallback to original logic if RAG doesn't find anything
        if hasattr(self, 'past_good_actions'):
            similar_state = None
            highest_similarity = 0.0
            for past_state in self.past_good_actions:
                similarity = difflib.SequenceMatcher(None, current_state_str, past_state).ratio()
                if similarity > 0.85 and similarity > highest_similarity:
                    highest_similarity = similarity
                    similar_state = past_state

            if similar_state:
                good_actions = self.past_good_actions[similar_state]
                if good_actions:
                    return True, f"Based on a similar past state, a good action taken was: {good_actions[0]}."

        return False, ""

    import re

    def _extract_goal_from_instructions(self, instructions: str) -> str | None:
        """Extract goal string from instructions using regex."""
        match = re.search(r"goal[:\s]+([\w\s]+)", instructions, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list) -> tuple:
        """
        Improved method with structured context and goal extraction for RAG.
        """
        self.states.append(observation.state.as_json())

        current_state_str = json.dumps(observation.state.as_json(), sort_keys=True)
        current_goal = self._extract_goal_from_instructions(self.instructions) or "default_goal"

        # Call RAG system for recommendations with current goal and memory buffer
        rag_found, retrieved_action_context = False, ""
        if self.rag_system:
            rag_found, retrieved_action_context = self.rag_system.get_action_recommendation(
                json.loads(current_state_str),
                k=3,
                goal=current_goal,
                recent_memory=memory_buf
            )

        # Format context with a clear header
        rag_context = """
RELEVANT PAST EXPERIENCES:
""" + (retrieved_action_context if rag_found else "No relevant past actions found.") + "\n" + "Based on this experience, consider similar actions for the current situation.\n"

        current_instructions = self.instructions + "\n" + rag_context

        status_prompt = create_status_from_state(observation.state)
        q1 = self.config['questions'][0]['text']
        q4 = self.config['questions'][3]['text']
        cot_prompt = self.config['prompts']['COT_PROMPT']
        memory_prompt = self.create_mem_prompt(memory_buf)

        repetitions = self.check_repetition(memory_buf)
        messages = [
            {"role": "user", "content": current_instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": q1},
        ]
        self.logger.info(f"Text sent to the LLM: {messages}")

        if self.use_self_consistency:
            response = self.get_self_consistent_response(messages, temp=repetitions/9, max_tokens=1024)
        else:
            response = self.openai_query(messages, max_tokens=1024)

        if self.use_reflection:
            reflection_prompt = [
                {
                    "role": "user",
                    "content": f"""
                    Instructions: {self.instructions}
                    Task: {q1}

                    Status: {status_prompt}
                    Memory: {memory_prompt}

                    Reasoning:
                    {response}

                    Is this reasoning valid given the Instructions, Status, and Memory?
                    - If YES, repeat it exactly.
                    - If NO, output the corrected reasoning only (no commentary).
                    """
                }
            ]
            response = self.openai_query(reflection_prompt, max_tokens=1024)

        if self.use_reasoning:
            response = self.remove_reasoning(response)
        self.logger.info(f"(Stage 1) Response from LLM: {response}")

        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": cot_prompt},
            {"role": "user", "content": response},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": q4},
        ]
        self.prompts.append(messages)


        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "AgentAction",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["ScanNetwork","ScanServices","ExploitService","FindData","ExfiltrateData"]
                        },
                        "parameters": { "type": "object" }  # any keys/values allowed
                    },
                    "required": ["action", "parameters"]
                }
            }
        }

        response = self.openai_query(messages, max_tokens=80, fmt=response_format)

        validated, error_msg = validate_responses.validate_agent_response(response)
        if validated is None:
            self.logger.info(f"Invalid response format: {response} - Error: {error_msg}")
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError:
                parsed_response = response

            response = json.dumps({
                "action": "InvalidResponse",
                "parameters": {
                    "error": error_msg,
                    "original": parsed_response
                }
            }, indent=2)

        if self.use_reasoning:
            response = self.remove_reasoning(response)

        self.responses.append(response)
        self.logger.info(f"(Stage 2) Response from LLM: {response}")
        print(f"(Stage 2) Response from LLM: {response}")
        return self.parse_response(response, observation.state)

    # Additional helper methods for RAG system
    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        if self.rag_system:
            return {
                'total_episodes': len(self.rag_system.past_episodes),
                'unique_good_states': len(self.rag_system.good_actions_by_state),
                'similarity_threshold': self.rag_system.similarity_threshold,
                'good_action_threshold': self.rag_system.good_action_threshold
            }
        return {'error': 'RAG system not initialized'}

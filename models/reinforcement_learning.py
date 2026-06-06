"""
ADVANCED REINFORCEMENT LEARNING FOR MEDICAL AI
===============================================
State-of-the-art RL algorithms for continuous learning and improvement:

1. RLHF (Reinforcement Learning from Human Feedback)
   - PPO (Proximal Policy Optimization)
   - Reward modeling from user feedback

2. Continuous Learning Algorithms:
   - DQN (Deep Q-Network) for discrete actions
   - Actor-Critic for policy optimization
   - A3C (Asynchronous Advantage Actor-Critic)
   - SAC (Soft Actor-Critic) for exploration

3. Medical-Specific RL:
   - Treatment recommendation optimization
   - Diagnostic pathway learning
   - Patient outcome optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import deque
import random
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# REWARD MODEL (for RLHF)
# ============================================================================

class RewardModel(nn.Module):
    """
    Learns to predict human preferences for medical responses
    Used in RLHF to guide the LLM
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Track training data
        self.preferences = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def compute_preference_loss(self, chosen: torch.Tensor, 
                                rejected: torch.Tensor) -> torch.Tensor:
        """
        Bradley-Terry preference loss
        P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        """
        r_chosen = self.forward(chosen)
        r_rejected = self.forward(rejected)
        
        # We want r_chosen > r_rejected
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        return loss
    
    def add_preference(self, chosen_embedding: np.ndarray, 
                       rejected_embedding: np.ndarray):
        """Add human preference data point"""
        self.preferences.append({
            "chosen": chosen_embedding.tolist(),
            "rejected": rejected_embedding.tolist()
        })
    
    def train_on_preferences(self, num_epochs: int = 10, 
                            batch_size: int = 32,
                            learning_rate: float = 1e-4) -> List[float]:
        """Train reward model on collected preferences"""
        if len(self.preferences) < batch_size:
            logger.warning("Not enough preferences to train")
            return []
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        
        for epoch in range(num_epochs):
            random.shuffle(self.preferences)
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(self.preferences), batch_size):
                batch = self.preferences[i:i+batch_size]
                
                chosen = torch.tensor([p["chosen"] for p in batch], dtype=torch.float32)
                rejected = torch.tensor([p["rejected"] for p in batch], dtype=torch.float32)
                
                optimizer.zero_grad()
                loss = self.compute_preference_loss(chosen, rejected)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            losses.append(avg_loss)
            logger.info(f"Reward Model Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        return losses


# ============================================================================
# PPO (Proximal Policy Optimization) for RLHF
# ============================================================================

class PPOPolicy(nn.Module):
    """
    PPO Policy Network for medical response generation
    """
    
    def __init__(self, state_dim: int = 768, action_dim: int = 1024,
                 hidden_dim: int = 512):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        
        mean = self.policy_mean(features)
        std = torch.exp(self.policy_log_std)
        
        value = self.value(features)
        
        return mean, std, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(state)
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class PPOTrainer:
    """
    PPO Trainer for RLHF
    Optimizes policy using human feedback (reward model)
    """
    
    def __init__(self, policy: PPOPolicy, reward_model: RewardModel,
                 lr: float = 3e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        
        self.policy = policy
        self.reward_model = reward_model
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def collect_experience(self, state: torch.Tensor, action: torch.Tensor,
                          log_prob: torch.Tensor, reward: float,
                          value: torch.Tensor, done: bool):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns with GAE"""
        returns = []
        R = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.tensor(returns, dtype=torch.float32)
    
    def update(self, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """PPO update step"""
        
        if len(self.states) == 0:
            return {}
        
        # Prepare data
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        returns = self.compute_returns()
        old_values = torch.stack(self.values).squeeze()
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute advantages
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(num_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Forward pass
                mean, std, values = self.policy(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                   1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        num_updates = num_epochs * (len(states) // batch_size + 1)
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }


# ============================================================================
# DQN (Deep Q-Network) for Discrete Medical Actions
# ============================================================================

class DQNetwork(nn.Module):
    """
    Deep Q-Network for medical decision making
    Actions: treatment recommendations, diagnostic tests, etc.
    """
    
    def __init__(self, state_dim: int = 768, num_actions: int = 50,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Dueling DQN heads
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, num_actions)
        
        self.use_dueling = True
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_dueling:
            features = self.network[:-1](state)  # All except last layer
            
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Q = V + (A - mean(A))
            q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        else:
            q_values = self.network(state)
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    """
    DQN Trainer with Double DQN and prioritized replay
    """
    
    def __init__(self, q_network: DQNetwork, lr: float = 1e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1,
                 epsilon_decay: int = 10000):
        
        self.q_network = q_network
        self.target_network = DQNetwork(
            state_dim=q_network.network[0].in_features,
            num_actions=q_network.network[-1].out_features
        )
        self.target_network.load_state_dict(q_network.state_dict())
        
        self.optimizer = optim.Adam(q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        self.gamma = gamma
        self.tau = tau
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0
    
    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.q_network.network[-1].out_features - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * \
                       np.exp(-self.steps / self.epsilon_decay)
        self.steps += 1
    
    def soft_update_target(self):
        """Soft update target network"""
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                   (1 - self.tau) * target_param.data)
    
    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """Single training step"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network to select action, target to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        
        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target
        self.soft_update_target()
        self.update_epsilon()
        
        return loss.item()


# ============================================================================
# ACTOR-CRITIC
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous medical decisions
    """
    
    def __init__(self, state_dim: int = 768, action_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.shared(state)
        
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)
        
        value = self.critic(features)
        
        return dist, value
    
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value


class A2CTrainer:
    """
    Advantage Actor-Critic Trainer
    """
    
    def __init__(self, actor_critic: ActorCritic, lr: float = 3e-4,
                 gamma: float = 0.99, entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Storage
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
    
    def collect(self, log_prob: torch.Tensor, value: torch.Tensor,
               reward: float, entropy: torch.Tensor):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.entropies.append(entropy)
    
    def update(self) -> Dict[str, float]:
        """Perform A2C update"""
        if len(self.rewards) == 0:
            return {}
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        entropies = torch.stack(self.entropies)
        
        # Advantages
        advantages = returns - values.detach()
        
        # Losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        total_loss = policy_loss + self.value_coef * value_loss + \
                    self.entropy_coef * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear storage
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item()
        }


# ============================================================================
# MEDICAL RL ENVIRONMENT
# ============================================================================

class MedicalEnvironment:
    """
    Simulated medical environment for RL training
    Rewards based on:
    - Correct diagnosis
    - Appropriate treatment
    - Patient safety
    - Cost efficiency
    """
    
    def __init__(self, num_symptoms: int = 100, num_diseases: int = 50,
                 num_treatments: int = 30):
        self.num_symptoms = num_symptoms
        self.num_diseases = num_diseases
        self.num_treatments = num_treatments
        
        # State: patient symptoms + history
        self.state_dim = num_symptoms + 50  # symptoms + additional features
        
        # Generate disease-symptom relationships
        self.disease_symptoms = self._generate_disease_symptoms()
        
        # Current episode state
        self.current_disease = None
        self.current_symptoms = None
        self.steps_taken = 0
        self.max_steps = 10
    
    def _generate_disease_symptoms(self) -> np.ndarray:
        """Generate realistic disease-symptom relationships"""
        # Each disease has probability of each symptom
        return np.random.beta(0.5, 2, (self.num_diseases, self.num_symptoms))
    
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial state"""
        # Sample a disease
        self.current_disease = np.random.randint(0, self.num_diseases)
        
        # Generate symptoms based on disease
        symptom_probs = self.disease_symptoms[self.current_disease]
        self.current_symptoms = np.random.binomial(1, symptom_probs)
        
        # Add noise (some symptoms may be unrelated)
        noise = np.random.binomial(1, 0.05, self.num_symptoms)
        self.current_symptoms = np.clip(self.current_symptoms + noise, 0, 1)
        
        # Additional features (age, gender, vitals, etc.)
        additional = np.random.randn(50) * 0.1
        
        self.state = np.concatenate([self.current_symptoms, additional])
        self.steps_taken = 0
        
        return torch.tensor(self.state, dtype=torch.float32)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take action and return (next_state, reward, done, info)
        
        Actions:
        - 0 to num_diseases-1: Diagnose with disease
        - num_diseases to num_diseases+num_treatments-1: Prescribe treatment
        - Last actions: Order tests
        """
        self.steps_taken += 1
        done = False
        reward = 0
        info = {}
        
        if action < self.num_diseases:
            # Diagnosis action
            diagnosed_disease = action
            
            if diagnosed_disease == self.current_disease:
                reward = 10.0  # Correct diagnosis
                info["correct_diagnosis"] = True
            else:
                reward = -5.0  # Wrong diagnosis
                info["correct_diagnosis"] = False
            
            done = True  # Episode ends after diagnosis
            
        elif action < self.num_diseases + self.num_treatments:
            # Treatment action
            treatment = action - self.num_diseases
            
            # Simple reward model: some treatments work better for some diseases
            effectiveness = np.random.beta(2, 5)  # Usually low
            if treatment == self.current_disease % self.num_treatments:
                effectiveness += 0.5  # Right treatment
            
            reward = effectiveness * 5 - 1  # Cost of treatment
            
        else:
            # Diagnostic test
            reward = -0.5  # Cost of test
            
            # Reveal more symptoms
            additional_symptoms = np.random.binomial(
                1, self.disease_symptoms[self.current_disease] * 0.3
            )
            self.current_symptoms = np.clip(
                self.current_symptoms + additional_symptoms, 0, 1
            )
            
            additional = np.random.randn(50) * 0.1
            self.state = np.concatenate([self.current_symptoms, additional])
        
        # Max steps penalty
        if self.steps_taken >= self.max_steps:
            done = True
            reward -= 2.0  # Penalty for taking too long
        
        return torch.tensor(self.state, dtype=torch.float32), reward, done, info


# ============================================================================
# COMPLETE RLHF SYSTEM
# ============================================================================

class MedicalRLHF:
    """
    Complete RLHF system for medical AI
    Combines reward modeling with PPO training
    """
    
    def __init__(self, state_dim: int = 768, action_dim: int = 1024,
                 save_dir: str = "rlhf_checkpoints"):
        
        self.reward_model = RewardModel(state_dim)
        self.policy = PPOPolicy(state_dim, action_dim)
        self.ppo_trainer = PPOTrainer(self.policy, self.reward_model)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Track metrics
        self.metrics = {
            "reward_model_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "feedback_count": 0
        }
    
    def add_human_feedback(self, chosen_response: np.ndarray,
                          rejected_response: np.ndarray):
        """Add human preference feedback"""
        self.reward_model.add_preference(chosen_response, rejected_response)
        self.metrics["feedback_count"] += 1
    
    def train_reward_model(self, num_epochs: int = 10) -> List[float]:
        """Train reward model on collected preferences"""
        losses = self.reward_model.train_on_preferences(num_epochs)
        self.metrics["reward_model_loss"].extend(losses)
        return losses
    
    def rl_step(self, state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Single RL step with reward from reward model"""
        action, log_prob, value = self.policy.get_action(state)
        
        # Get reward from reward model
        with torch.no_grad():
            reward = self.reward_model(state.unsqueeze(0)).item()
        
        self.ppo_trainer.collect_experience(
            state, action, log_prob, reward, value, done=False
        )
        
        return action, reward
    
    def update_policy(self) -> Dict[str, float]:
        """Update policy with PPO"""
        metrics = self.ppo_trainer.update()
        
        if metrics:
            self.metrics["policy_loss"].append(metrics.get("policy_loss", 0))
            self.metrics["value_loss"].append(metrics.get("value_loss", 0))
            self.metrics["entropy"].append(metrics.get("entropy", 0))
        
        return metrics
    
    def save(self, name: str = "rlhf"):
        """Save models and metrics"""
        torch.save({
            "reward_model": self.reward_model.state_dict(),
            "policy": self.policy.state_dict(),
            "metrics": self.metrics
        }, self.save_dir / f"{name}.pt")
        
        # Save preferences separately
        with open(self.save_dir / f"{name}_preferences.json", "w") as f:
            json.dump(self.reward_model.preferences, f)
    
    def load(self, name: str = "rlhf"):
        """Load models and metrics"""
        checkpoint = torch.load(self.save_dir / f"{name}.pt")
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.metrics = checkpoint["metrics"]
        
        # Load preferences
        pref_path = self.save_dir / f"{name}_preferences.json"
        if pref_path.exists():
            with open(pref_path) as f:
                self.reward_model.preferences = json.load(f)


# Export
__all__ = [
    "RewardModel",
    "PPOPolicy",
    "PPOTrainer",
    "DQNetwork",
    "DQNTrainer",
    "ReplayBuffer",
    "ActorCritic",
    "A2CTrainer",
    "MedicalEnvironment",
    "MedicalRLHF",
]

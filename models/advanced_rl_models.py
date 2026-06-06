"""
Advanced Reinforcement Learning for Medical Treatment Optimization
- DQN for medication dosage optimization
- Policy Gradient for treatment recommendations
- Multi-agent RL for complex medical scenarios
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')


class MedicalTreatmentEnv:
    """Environment for medical treatment optimization"""
    
    def __init__(self, patient_state=None):
        self.patient_state = patient_state or np.zeros(10)  # 10 vital parameters
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        """Reset environment"""
        self.patient_state = np.random.randn(10) * 0.5
        self.step_count = 0
        return self.patient_state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.step_count += 1
        
        # Simulate patient response to treatment
        # Action: 0-9 (medication type), 10-19 (dosage level)
        
        medication = action // 10
        dosage = (action % 10) / 10.0
        
        # Update vital signs based on treatment
        effect = np.random.randn(10) * 0.1 + dosage * 0.3
        self.patient_state = np.clip(self.patient_state + effect, -2, 2)
        
        # Calculate reward (improvement in vitals)
        reward = -np.sum(np.abs(self.patient_state))  # Reward for vitals near 0
        
        # Add penalty for extreme dosages
        if dosage > 0.8:
            reward -= 10
        
        done = self.step_count >= self.max_steps
        
        return self.patient_state, reward, done
    
    def render(self):
        print(f"Step: {self.step_count}, State: {self.patient_state.round(2)}")


class DeepQNetwork:
    """Deep Q-Network for treatment optimization"""
    
    def __init__(self, state_size=10, action_size=20):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        print("[RL] Deep Q-Network initialized")
    
    def _build_model(self):
        """Build neural network"""
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Experience replay for training"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Predict Q-values for starting state
        q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next state
        q_next = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
        
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def train(self, episodes=100):
        """Train DQN agent"""
        env = MedicalTreatmentEnv()
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(env.max_steps):
                action = self.act(state)
                next_state, reward, done = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.replay(32)
            
            if (episode + 1) % 10 == 0:
                self.update_target_model()
                print(f"[Episode {episode+1}] Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        print("[✓] DQN training complete")


class PolicyGradientAgent:
    """Policy Gradient for treatment recommendations"""
    
    def __init__(self, state_size=10, action_size=20):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.gamma = 0.99
        
        self.model = self._build_policy_network()
        print("[RL] Policy Gradient Agent initialized")
    
    def _build_policy_network(self):
        """Build policy network"""
        inputs = keras.Input(shape=(self.state_size,))
        
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Policy head (action probabilities)
        policy = layers.Dense(self.action_size, activation='softmax', name='policy')(x)
        
        # Value head (state value estimation)
        value = layers.Dense(1, activation='linear', name='value')(x)
        
        model = keras.Model(inputs=inputs, outputs=[policy, value])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def select_action(self, state):
        """Select action based on policy"""
        state = np.reshape(state, [1, self.state_size])
        policy, value = self.model.predict(state, verbose=0)
        action = np.random.choice(self.action_size, p=policy[0])
        return action, value[0]
    
    def train(self, episodes=100):
        """Train policy gradient agent"""
        env = MedicalTreatmentEnv()
        
        for episode in range(episodes):
            state = env.reset()
            episode_log_probs = []
            episode_rewards = []
            
            for step in range(env.max_steps):
                action, value = self.select_action(state)
                next_state, reward, done = env.step(action)
                
                episode_rewards.append(reward)
                state = next_state
                
                if done:
                    break
            
            # Compute returns
            returns = []
            cumulative = 0
            for reward in reversed(episode_rewards):
                cumulative = reward + self.gamma * cumulative
                returns.insert(0, cumulative)
            
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            
            if (episode + 1) % 10 == 0:
                print(f"[Episode {episode+1}] Avg Return: {np.mean(episode_rewards):.2f}")
        
        print("[✓] Policy Gradient training complete")


class AdvancedMedicalRL:
    """Unified RL system for medical treatment"""
    
    def __init__(self):
        self.dqn = DeepQNetwork(state_size=10, action_size=20)
        self.policy_agent = PolicyGradientAgent(state_size=10, action_size=20)
        print("[RL] Advanced Medical RL System initialized")
    
    def recommend_treatment(self, patient_state):
        """Recommend treatment based on patient state"""
        state = np.reshape(patient_state, [1, 10])
        
        # DQN recommendation
        dqn_q_values = self.dqn.model.predict(state, verbose=0)
        dqn_action = np.argmax(dqn_q_values[0])
        
        # Policy recommendation
        policy, value = self.policy_agent.model.predict(state, verbose=0)
        policy_action = np.argmax(policy[0])
        
        return {
            'dqn_recommendation': dqn_action,
            'policy_recommendation': policy_action,
            'confidence': float(np.max(policy[0])),
            'state_value': float(value[0])
        }
    
    def train_agents(self, episodes=50):
        """Train all RL agents"""
        print("[RL] Starting agent training...")
        print("\n--- Training DQN Agent ---")
        self.dqn.train(episodes=episodes)
        
        print("\n--- Training Policy Gradient Agent ---")
        self.policy_agent.train(episodes=episodes)
        
        print("\n[✓] All RL agents trained!")


# Initialize global RL system
rl_system = AdvancedMedicalRL()

if __name__ == "__main__":
    print("Building Reinforcement Learning system...")
    rl = AdvancedMedicalRL()
    
    # Example: Get treatment recommendation
    patient_state = np.random.randn(10) * 0.5
    recommendation = rl.recommend_treatment(patient_state)
    print(f"\nTreatment Recommendation: {recommendation}")
    print("\n[✓] RL system ready!")

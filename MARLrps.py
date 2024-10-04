import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pygame
import sys

# Initialize Pygame
pygame.init()

# Define constants(actually not)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30
ACTIONS = ["rock", "paper", "scissors"]
NUM_AGENTS = int(input("Write the total number of agents "))
NUM_ROUNDS = int(input("Write the number of rounds "))
BATCH_SIZE = 32
GAMMA = 0.95  # Discount factor
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Load images
ROCK_IMAGE = pygame.image.load("rock.png")
PAPER_IMAGE = pygame.image.load("paper.png")
SCISSORS_IMAGE = pygame.image.load("scissors.png")
BACKGROUND_IMAGE = pygame.image.load("rock_paper_scissors_background.png")

# Set up the Pygame window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RPS-MARL-Agents")
font = pygame.font.Font(None, 36)

# Function to determine the winner of the game
def get_winner(action1, action2):
    if action1 == action2:
        return 0  # Draw
    elif (action1 == "rock" and action2 == "scissors") or \
         (action1 == "paper" and action2 == "rock") or \
         (action1 == "scissors" and action2 == "paper"):
        return 1  # Action1 wins
    else:
        return -1  # Action2 wins

# DQN Agent class
class DQNAgent:
    def __init__(self, name):
        self.name = name
        self.state_size = len(ACTIONS)
        self.action_size = len(ACTIONS)
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def act(self):
        if np.random.rand() <= self.epsilon:  # Explore
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.zeros((1, self.state_size)))  # Placeholder state
        return np.argmax(act_values[0])  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

# Rock-Paper-Scissors Environment
class RockPaperScissorsEnv:
    def __init__(self, agents):
        self.agents = agents
    
    def play_round(self):
        actions = {agent.name: agent.act() for agent in self.agents}
        print(f"Actions: {actions}")

        rewards = {agent.name: 0 for agent in self.agents}
        
        # Determine winners and assign rewards
        for agent in self.agents:
            action = actions[agent.name]
            for opponent in self.agents:
                if opponent != agent:
                    opponent_action = actions[opponent.name]
                    winner = get_winner(ACTIONS[action], ACTIONS[opponent_action])
                    if winner == 1:  # Agent wins
                        rewards[agent.name] += 1
                    elif winner == -1:  # Opponent wins
                        rewards[opponent.name] += 1

        # Store experiences
        for agent in self.agents:
            action = actions[agent.name]
            state = np.zeros((1, agent.state_size))  # Placeholder state
            next_state = np.zeros((1, agent.state_size))  # Placeholder next state
            done = False  # Assuming game is not done
            agent.remember(state, action, rewards[agent.name], next_state, done)

    def play_game(self, rounds=NUM_ROUNDS):
        for _ in range(rounds):
            self.play_round()
            for agent in self.agents:
                agent.replay()

# Pygame Visualization
def draw_agents(agents, round_number):
    screen.blit(BACKGROUND_IMAGE, 0, 0)

    # Display round number
    round_text = font.render(f"Round: {round_number + 1}", True, (0, 0, 0))
    screen.blit(round_text, (10, 10))

    # Display each agent's action and corresponding image
    for i, agent in enumerate(agents):
        action_index = agent.act()
        action_text = font.render(f"{agent.name}: {ACTIONS[action_index]}", True, (0, 0, 0))
        screen.blit(action_text, (10, 50 + i * 40))

        # Draw the corresponding action image
        if ACTIONS[action_index] == "rock":
            screen.blit(ROCK_IMAGE, (300, 50 + i * 40))
        elif ACTIONS[action_index] == "paper":
            screen.blit(PAPER_IMAGE, (300, 50 + i * 40))
        elif ACTIONS[action_index] == "scissors":
            screen.blit(SCISSORS_IMAGE, (300, 50 + i * 40))

    pygame.display.flip()  # Update the screen

def main():
    agents = [DQNAgent(name=f"Agent_{i}") for i in range(NUM_AGENTS)]
    env = RockPaperScissorsEnv(agents)

    clock = pygame.time.Clock()

    for round_number in range(NUM_ROUNDS):
        env.play_round()
        draw_agents(agents, round_number)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for agent in agents:
            agent.replay()

        clock.tick(FPS)  # Control the frame rate

    pygame.quit()

# Run the simulation
if __name__ == "__main__":
    main()

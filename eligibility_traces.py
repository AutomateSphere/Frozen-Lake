# Eligibility Traces implementation
# Frozen lake

import gym
import numpy as np
import matplotlib.pyplot as plt


def epsilon_greedy_policy(Q, epsilon, n_actions):
    def policy_fn(state):
        if np.random.rand() < epsilon:
            return np.random.choice(n_actions)
        else:
            return np.argmax(Q[state])

    return policy_fn


def run_eligibility_trace(num_episodes, grid_size, is_slippery=False):
    # Set up environment
    env = gym.make('FrozenLake-v1', map_name=f"{grid_size}x{grid_size}", render_mode="ansi", is_slippery=is_slippery)
    env.reset()
    env.render()

    # default values
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    lambda_val = 0.9

    alpha = 0.1  # Initial learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.999  # Exploration rate decay
    lambda_val = 0.9  # Eligibility trace decay

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # Initialize Q-Table with zeros
    Q = np.zeros((n_states, n_actions))
    eligibility_trace = np.zeros((n_states, n_actions))

    episode_rewards = []
    steps_per_episode = []

    # eligibility_traces learning
    for episode in range(num_episodes):
        done = False  # variable to check if the agent fall down to holes or finish the maze
        total_reward = 0
        num_steps = 0  # Storing number of steps for each iteration

        state, probability = env.reset()  # starting state
        policy = epsilon_greedy_policy(Q, epsilon, n_actions)
        action = policy(state)

        while not done:
            num_steps += 1  # count the steps
            action = epsilon_greedy_policy(Q, epsilon, n_actions)(state)
            next_state, reward, done, _, _ = env.step(action)

            next_action = policy(next_state)

            td_error = reward + gamma * Q[next_state, next_action] - Q[state, action]

            eligibility_trace[state, action] += 1

            Q += alpha * td_error * eligibility_trace
            eligibility_trace *= gamma * lambda_val

            total_reward += reward
            state = next_state
            # action = next_action

            # if the agent fell down to any holes or finished, exit that iteration
            if done:
                # steps_per_episode.append(num_steps)
                print(f'Episode: {episode + 1}, Reward: {reward}, Number of steps: {num_steps}')
                break

        episode_rewards.append(total_reward)
        steps_per_episode.append(num_steps)

        # Update exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    env.close()
    return Q, episode_rewards, steps_per_episode

def plotting_graph(grid_size, total_rewards, steps_per_episode):
    # Calculating average rewards
    avg_rewards = [np.mean(total_rewards[max(0, i - 100):i + 1]) for i in range(len(total_rewards))]
    # Plot average rewards over episodes
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward over Episodes for {grid_size}x{grid_size} (Eligibility Traces)')
    plt.legend()
    plt.grid(True)
    plt.savefig('avg_reward_over_episode_ET.pdf')
    plt.show()

    # Plot steps per episode
    avg_steps_per_episode = [sum(steps_per_episode[:i + 1]) / len(steps_per_episode[:i + 1]) for i in
                             range(len(steps_per_episode))]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(avg_steps_per_episode) + 1), avg_steps_per_episode)
    plt.title(f'Average Steps per episode for {grid_size}x{grid_size} (Eligibility Traces)')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.savefig('step_per_episode_ET.pdf')
    plt.grid(True)
    plt.show()

    # Learning curve
    plt.figure(figsize=(10, 6))
    # Calculating average rewards
    avg_reward_per_episode = [sum(total_rewards[:i + 1]) / len(total_rewards[:i + 1]) for i in
                   range(len(total_rewards))]
    plt.plot(range(1, len(avg_reward_per_episode) + 1), avg_reward_per_episode)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'Learning Curve for {grid_size}x{grid_size} environment Over 100,000 Iterations (Eligibility Traces)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve_ET.pdf')
    plt.show()
    plt.close()


if __name__ == "__main__":
    grid_size = int(input("Enter the grid size (e.g., 4 for 4x4): "))
    episodes = int(input("Enter total number of episodes: "))

    # running model
    Q, rewards, steps = run_eligibility_trace(episodes, grid_size, is_slippery=False)
    # plotting graph
    plotting_graph(grid_size, rewards, steps)
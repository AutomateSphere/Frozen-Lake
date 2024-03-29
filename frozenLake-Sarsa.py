# SARSA implementation
# Frozen lake
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Plotting function to visualize alpha schedule
def plot_schedule(schedule, ylab='Alpha'):
    plt.figure(figsize=(10, 6))
    plt.plot(schedule, label='Alpha Schedule')
    plt.xlabel('Episodes')
    plt.ylabel(ylab)
    plt.title(f'{ylab} Schedule')
    plt.legend()
    plt.show()

def plot_sd(rewards, episodes):
    # Calculate average reward and standard deviation
    avg_reward = np.mean(rewards)
    std_dev = np.std(rewards)

    # Plotting the average reward curve with error bars representing standard deviation
    plt.errorbar(episodes, avg_reward, yerr=std_dev, fmt='-o', label='Average Reward')

    # Add labels and legend
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward with Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting function to visualize Q-table and average rewards
def draw_plot(total_rewards, Q, grid_size, steps_per_episode, rewards_per_episode):


    # Q-table visualization
    df = pd.DataFrame(Q)
    q_table_df = df.applymap(lambda x: f'{x:.3f}')
    action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
    q_table_df.rename(columns=action_names, inplace=True)
    state_labels = [(i, j) for i in range(int(grid_size)) for j in range(int(grid_size))]
    q_table_df.index = state_labels

    # Saving the Q-table DataFrame to a PDF file
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    column_widths = [0.15] * len(q_table_df.columns)
    # Add a table to the axis
    table = ax.table(cellText=q_table_df.values,
                     colLabels=q_table_df.columns,
                     rowLabels=q_table_df.index,
                     loc='center',
                     cellLoc='center',
                     colWidths=column_widths)
    # Set font size and alignment for better readability
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')  # Set heading to bold

    plt.title('Q-table Visualization for SARSA Learning')
    plt.savefig('q_table_SARSA.pdf')

    # Plot Q-table
    plt.figure(figsize=(10, 8))
    sns.heatmap(Q, cmap="YlGnBu", annot=True, fmt=".2f", cbar=False)
    plt.title(f'Heatmap of Q-values for {grid_size}x{grid_size} environment\nBrighter colors indicate higher values')
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.title("Heatmap of Q-values")
    plt.show()

    # Plot average rewards over episodes
    plt.figure(figsize=(10, 6))
    # Calculating average rewards
    avg_rewards = [np.mean(total_rewards[max(0, i - 100):i + 1]) for i in range(len(total_rewards))]
    plt.plot(np.arange(len(avg_rewards)), avg_rewards, label='Average Reward')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Episodes (SARSA)')
    plt.legend()
    plt.grid(True)
    plt.savefig('average_reward_SARSA.pdf')
    plt.show()

    # Plot steps per episode
    avg_steps_per_episode = [sum(steps_per_episode[:i + 1]) / len(steps_per_episode[:i + 1]) for i in
                             range(len(steps_per_episode))]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(avg_steps_per_episode) + 1), avg_steps_per_episode)

    plt.title(f'Average Steps per episode for {grid_size}x{grid_size} environment (SARSA)')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.savefig('step_per_episode_SARSA.pdf')
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
    plt.title(f'Learning Curve for {grid_size}x{grid_size} environment Over 100,000 Iterations (SARSA)')
    plt.legend()
    plt.grid(True)
    plt.savefig('average_reward_SARSA.pdf')
    plt.show()
    plt.close()

def plot_standard_deviation(episodes, avg_rewards, std_devs):
    plt.errorbar(range(episodes), avg_rewards, yerr=std_devs, fmt='o', capsize=3, capthick=1, markeredgecolor='blue', label='SARSA')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward with Standard Deviation (SARSA)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

def run_sarsa(episodes, grid_size, is_slippery=False):
    # Set up environment
    env = gym.make('FrozenLake-v1', map_name=f"{grid_size}x{grid_size}", render_mode="ansi", is_slippery=is_slippery)
    env.reset()
    env.render()

    # Initialize Q-Table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Define hyperparameters
    alpha = 0.05
    gamma = 0.9
    egreedy = 0.7
    epsilon = 0.1
    egreedy_decay = 0.999

    # List to store total rewards for each episode
    total_rewards = []
    steps_per_episode = []
    rewards_per_episode = []
    avg_rewards = []
    std_devs = []

    # SARSA Learning
    for i in range(episodes):
        total_reward = 0
        num_steps = 0  # Storing number of steps for each iteration
        state, probability = env.reset() # starting state
        done = False # variable to check if the agent fall down to holes or finish the maze

        while not done:
            num_steps += 1 # count the steps

            # getting action for current state
            random_egreedy = np.random.random()

            if random_egreedy > egreedy:
                random_values = Q[state] + np.random.rand(1, env.action_space.n) / 1000
                action = np.argmax(random_values, axis=1)[0]
                action = action.item()
            else:
                # get action
                action = env.action_space.sample()

            if egreedy > epsilon:
                egreedy *= egreedy_decay

            # getting next state
            next_state, reward, done, _, _ = env.step(action)

            # get action for next state
            if random_egreedy > egreedy:
                random_values = Q[state] + np.random.rand(1, env.action_space.n) / 1000
                _action = np.argmax(random_values, axis=1)[0]
                _action = _action.item()
            else:
                # get action
                _action = env.action_space.sample()

            if egreedy > epsilon:
                egreedy *= egreedy_decay

            Q[state, action] += alpha * (reward + gamma * Q[next_state, _action] - Q[state, action])

            # move on to next state
            state = next_state
            action = _action
            total_reward += reward
            # if the agent fell down to any holes or finished, exit that iteration
            if done:
                total_rewards.append(total_reward)
                steps_per_episode.append(num_steps)
                rewards_per_episode.append(total_reward)
                print(f'Episode: {i + 1}, Reward: {reward}, Number of steps: {num_steps}')
                break

        avg_reward = np.mean(total_rewards)
        std_dev = np.std(total_rewards)
        avg_rewards.append(avg_reward)
        std_devs.append(std_dev)

    env.close()

    return total_rewards, steps_per_episode, rewards_per_episode, Q, std_devs, avg_rewards

if __name__ == '__main__':
    grid_size = int(input("Enter the grid size (e.g., 4 for 4x4): "))
    episodes = int(input("Enter total number of episodes: "))
    #running model
    total_rewards, steps_per_episode, rewards_per_episode, Q, std_devs, avg_rewards = run_sarsa(episodes, grid_size, is_slippery=False)

    #plotting graph
    #draw_plot(total_rewards, Q, grid_size, steps_per_episode, rewards_per_episode)
    #plot_sd(total_rewards, episodes)
    print(np.std(std_devs), np.mean(avg_rewards))
    plot_standard_deviation(episodes, avg_rewards, std_devs)



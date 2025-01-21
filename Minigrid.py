import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper
import time

# Charger l'environnement Minigrid
env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0")

# Définir les hyperparamètres
num_episodes = 1000
alpha = 0.5   # Taux d'apprentissage
gamma = 0.99  # Facteur de discount
epsilon = 0.5  # Paramètre d'exploration (epsilon-greedy)

def flatten_state(obs):
    """Transforme une observation en un état unique (hashable)."""
    if isinstance(obs, dict) and "image" in obs:  # Observation complète
        return obs["image"].flatten().tobytes()
    elif isinstance(obs, np.ndarray):  # Si c'est un tableau numpy
        return obs.flatten().tobytes()
    else:
        raise ValueError(f"Format d'observation inattendu : {type(obs)}")

def epsilon_greedy_action(Q, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q.get(state, np.zeros(n_actions)))

def train_sarsa(env, num_episodes, alpha, gamma, epsilon):
    Q = {}  # Utilisation d'un dictionnaire pour représenter Q
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()  # Récupérer uniquement l'observation
        state = flatten_state(obs)
        action = epsilon_greedy_action(Q, state, env.action_space.n, epsilon)
        total_reward = 0

        while True:
            obs, reward, done, _, _ = env.step(action)  # Extraire observation et autres valeurs
            total_reward += reward

            next_state = flatten_state(obs)
            next_action = epsilon_greedy_action(Q, next_state, env.action_space.n, epsilon)

            if state not in Q:
                Q[state] = np.zeros(env.action_space.n)
            if next_state not in Q:
                Q[next_state] = np.zeros(env.action_space.n)

            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )

            state, action = next_state, next_action

            if done:
                break

        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)
        alpha = max(0.1, alpha * 0.995)
    return Q, rewards

def train_qlearning(env, num_episodes, alpha, gamma, epsilon):
    Q = {}  # Utilisation d'un dictionnaire pour représenter Q
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()  # Récupérer uniquement l'observation
        state = flatten_state(obs)
        total_reward = 0

        while True:
            action = epsilon_greedy_action(Q, state, env.action_space.n, epsilon)
            obs, reward, done, _, _ = env.step(action)  # Extraire observation et autres valeurs
            total_reward += reward

            next_state = flatten_state(obs)
            next_action = epsilon_greedy_action(Q, next_state, env.action_space.n, epsilon) 

            if state not in Q:
                Q[state] = np.zeros(env.action_space.n)
            if next_state not in Q:
                Q[next_state] = np.zeros(env.action_space.n)


            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[next_state][next_action]) - Q[state][action]
            )

            state = next_state
            
            if done:
                break

        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)
        alpha = max(0.1, alpha * 0.995)
    return Q, rewards

def visualize_policy(env, Q, delay=0.25, max_steps=100):
    """
    Visualise un parcours dans l'environnement en suivant une politique donnée.
    """
    # Recréer l'environnement avec un mode de rendu
    env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0", render_mode="human")

    obs, _ = env.reset()  # Réinitialiser l'environnement
    state = flatten_state(obs)
    env.render()
    time.sleep(delay)

    for step in range(max_steps):
        # Obtenir la meilleure action selon Q
        if state in Q:
            action = np.argmax(Q[state])
        else:
            print("État inconnu, l'agent effectue une action aléatoire.")
            action = env.action_space.sample()

        # Exécuter l'action
        obs, reward, done, _, _ = env.step(action)
        state = flatten_state(obs)

        # Afficher la grille
        env.render()  # Afficher sans "mode"
        print(f"Étape {step + 1}: Récompense = {reward}")
        time.sleep(delay)

        if done:
            print("L'agent a atteint la fin de l'épisode.")
            break
    else:
        print("Nombre maximum d'étapes atteint.")

# Entraînement
Q_sarsa, rewards_sarsa = train_sarsa(env, num_episodes, alpha, gamma, epsilon)
Q_qlearning, rewards_qlearning = train_qlearning(env, num_episodes, alpha, gamma, epsilon)



# Visualiser un parcours en suivant la politique apprise par SARSA
print("Visualisation du parcours avec SARSA :")
visualize_policy(env, Q_sarsa, delay=0.25)

# Visualiser un parcours en suivant la politique apprise par Q-Learning
print("Visualisation du parcours avec Q-Learning :")
visualize_policy(env, Q_qlearning, delay=0.25)

# Visualisation des résultats
plt.plot(rewards_sarsa, label="SARSA")
plt.plot(rewards_qlearning, label="Q-Learning")
plt.xlabel("Épisode")
plt.ylabel("Gain cumulatif")
plt.legend()
plt.title("Comparaison des performances : SARSA vs Q-Learning")
plt.grid()
plt.show()

#np.save('projet/reward_sarsa',rewards_sarsa)
#np.save('projet/reward_qlearning',rewards_qlearning)

def moving_average(data, window_size):
    """
    Calcule la moyenne glissante d'une liste de données.
    :param data: Liste des données (ex: récompenses par épisode)
    :param window_size: Taille de la fenêtre de moyenne
    :return: Liste des moyennes glissantes
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Calcul des moyennes glissantes (fenêtre de 50 épisodes)
window_size = 50
rewards_sarsa_ma = moving_average(rewards_sarsa, window_size)
rewards_qlearning_ma = moving_average(rewards_qlearning, window_size)

# Visualisation des résultats avec les moyennes glissantes
plt.figure(figsize=(10, 6))
plt.plot(rewards_sarsa, color='blue', alpha=0.3, label="Récompenses brutes (SARSA)")
plt.plot(rewards_qlearning, color='orange', alpha=0.3, label="Récompenses brutes (Q-Learning)")
plt.plot(range(window_size - 1, len(rewards_sarsa)), rewards_sarsa_ma, color='blue', label="Moyenne glissante (SARSA)")
plt.plot(range(window_size - 1, len(rewards_qlearning)), rewards_qlearning_ma, color='orange', label="Moyenne glissante (Q-Learning)")
plt.xlabel("Épisode")
plt.ylabel("Gain cumulatif")
plt.title("Évolution des récompenses avec moyennes glissantes (fenêtre = 50)")
plt.legend()
plt.grid()
plt.show()

np.save("reward_qlearning_8x8.npy",rewards_qlearning)
np.save("reward_sarsa_8x8.npy",rewards_sarsa)
from numpy import load
import matplotlib.pyplot as plt

def sarsa(): 
    reward_sasra=load('reward_sarsa.npy')

    plt.plot(reward_sasra,label="Sarsa")

    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()



def qlearning():
    reward_qlearning=load('reward_qlearning.npy')
    plt.plot(reward_qlearning,label="Q-learning")

    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()
    
qlearning()
sarsa()
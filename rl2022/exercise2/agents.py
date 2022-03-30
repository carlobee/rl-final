from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim

class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: int) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """

        # generate a random number between 0 and 1
        p = random.random()
        # dictionary which holds only values for this obs, {(this obs, act) : Q}
        filtered = {k: v for k, v in self.q_table.items() if k[0] == obs}

        # if number falls out of epsilon range, return best possible action for this state
        if p < (1 - self.epsilon) and bool(filtered) is not False:
            return max(filtered, key=filtered.get)[1]
        else:
            # return a random action
            return random.randint(0, self.n_acts-1)


    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class QLearningAgent(Agent):
    """
    Agent using the Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: int, action: int, reward: float, n_obs: int, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """

        sa_pair = (obs, action)

        # while we have not reached a terminal state
        if not done:
            next_max = 0
            # dictionary which holds only values for this obs, {(this obs, act) : Q}
            filtered = {k: v for k, v in self.q_table.items() if k[0] == n_obs}

            # if the values for this obv are not none
            if bool(filtered) is not False:
                # get the best value
                next_max = max(filtered.values())

            # perform the update to the q-table
            self.q_table[sa_pair] = (1-self.alpha) * self.q_table[sa_pair] + self.alpha * (reward + self.gamma * next_max)

        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        self.epsilon = 0.1
        self.alpha = 0.6


class MonteCarloAgent(Agent):
    """
    Agent using the Monte-Carlo algorithm for training

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(
        self, obses: List[int], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obses (List(int)): list of received observations representing environmental states
            of trajectory (in the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """

        # initialise updated Q-table
        updated_values = {}

        # for each observation in the episode
        for i in range(len(obses)):
            sa_pair = (obses[i], actions[i])
            # make sure we have not seen the sa-pair previously in this episode (for first-visit)
            if sa_pair not in updated_values.keys():
                # increment the global counter
                self.sa_counts[sa_pair] = self.sa_counts.get(sa_pair, 0) + 1
                # unpack the current Q-value for this sa_pair so we have the total rewards for all observations
                returns = self.q_table[sa_pair] * (self.sa_counts[sa_pair] - 1)
                # update the Q-value based on subsequent rewards + returns
                updated_values[sa_pair] = (returns + sum(rewards[i:]) * self.gamma) / self.sa_counts[sa_pair]

        # update master Q-table with all Q-value updates in this episode
        self.q_table.update(updated_values)
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        # schedule = {0: 0.9, 100000: 0.7, 300000: 0.6, 500000: 0.4, 700000: 0.3, 900000: 0.01, 950000: 0.001}
        #
        # if timestep in schedule:
        #     self.epsilon = schedule[timestep]

        max_epsilon = 1.0  # Exploration probability at start
        min_epsilon = 0.01  # Minimum exploration probability
        decay_rate = 0.000001  # Exponential decay rate for exploration prob
        e = 2.718281828459045

        self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * e**(-decay_rate * timestep)

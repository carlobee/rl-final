from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict

from gym.spaces import Space
from gym.spaces.utils import flatdim


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self) -> List[int]:
        """Chooses an action for all agents for stateless task

        :return (List[int]): index of selected action for each agent
        """
        ...

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


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """

        actions = []

        # for each agent
        for a in range(self.num_agents):
            # choose best action to exploit
            if random.random() < (1 - self.epsilon) and len(self.q_tables[a]) != 0:
                actions.append(max([(v, k) for k, v in self.q_tables[a].items()])[1])
            else:
                # else explore using random action
                actions.append(random.randint(0, self.n_acts[a] - 1))

        return actions

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current actions of each agent
        """
        updated_values = []

        # for each agent
        for a, q_table in enumerate(self.q_tables):
            # get the q value for the actual action taken
            Q = q_table[actions[a]]
            # get the q value for the best possible action
            Q_best = max(q_table.values())
            # calculate update
            Q_update = Q + self.learning_rate * (rewards[a] + self.gamma * Q_best - Q) * (1 - dones[a])
            # update q table
            q_table[actions[a]] = Q_update
            updated_values.append(Q_update)

        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        self.epsilon = 0.2
        self.learning_rate = 1e-3

class JointActionLearning(MultiAgent):
    """
    Agents using the Joint Action Learning algorithm with Opponent Modelling

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping joint actions ACTs
            to respective Q-values for all agents
        :attr models (List[DefaultDict]): each agent holding model of other agent
            mapping other agent actions to their counts

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 

    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        joint_action = []

        # list of agent dicts to hold {action : Q_vals}
        expected_value_list: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # for each of our agents
        for agent in range(self.num_agents):
            # for each action
            for action in range(self.n_acts[agent]):
                # for every other action, calculate expected value, the sum of joint Q-vals,
                # weighted by estimated probability joint actions
                total_C = 0
                for non_act in range(self.n_acts[1 - action]):
                    C = self.models[agent][(agent, non_act)]
                    total_C += self.models[agent][(action, non_act)]
                    expected_value_list[agent][action] += (C / max(1, total_C)) * self.q_tables[agent][
                        (action, non_act)]

            # choose best action to exploit
            if random.random() < (1 - self.epsilon):
                # get best action for each agent using EV list
                joint_action = (max(expected_value_list[agent], key=expected_value_list[agent].get) for a in
                                range(self.num_agents))
            else:
                # else explore using random action
                joint_action = [random.randint(0, self.n_acts[agent] - 1) for agent in range(self.num_agents)]

        return joint_action

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []

        # list of agent dicts to hold {action : Q_vals}
        expected_value_list: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # for each of our agents
        for agent in range(self.num_agents):
            # for each action
            for action in range(self.n_acts[agent]):
                # update the model
                self.models[agent][tuple(actions)] += 1

                # for every other action, calculate expected value, the sum of joint Q-vals,
                # weighted by estimated probability joint actions
                total_C = 0
                for non_act in range(self.n_acts[1 - action]):
                    C = self.models[agent][(agent, non_act)]
                    total_C += self.models[agent][(action, non_act)]
                    expected_value_list[agent][action] += (C / max(1, total_C)) * self.q_tables[agent][
                        (action, non_act)] * (dones[agent])

                # update Q-values
                self.q_tables[agent][tuple(actions)] = self.q_tables[agent][tuple(actions)] + self.learning_rate * (
                            rewards[agent] + self.gamma * max(expected_value_list[agent].values()) -
                            self.q_tables[agent][tuple(actions)])
                updated_values.append(self.q_tables[agent][tuple(actions)])

        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        self.epsilon = 0.2
        self.learning_rate = 1e-3

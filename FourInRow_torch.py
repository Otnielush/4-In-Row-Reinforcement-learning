import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import namedtuple, deque
import os.path as path

# little bug
model_path = 'D:\Programming\Jupyter\ReinforcementLearning'
model_path = path.join(model_path, 'FourInRow_animated_RL\Connect4_models\Connect4_model2_last.pt')


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

class ActorCritic(nn.Module):
    def __init__(self, name, training, obs_shape, act_shape, buffer_size, lr=1e-2):
        super(ActorCritic, self).__init__()
        self.obs_shape = obs_shape[::-1]
        self.name = name
        self.games_played = 0
        self.wins = 0

        self.conv1 = nn.Conv2d(2, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3)
        self.bn2 = nn.BatchNorm2d(12)
        self.maxpool1 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(self.calc_input_size(), 64)
        self.actor = nn.Linear(64, act_shape)
        self._training = training
        if self._training:
            self.history = deque(maxlen=1000000)
            self.critic = nn.Linear(64, 1)  # Critic is always 1
            self.saved_actions = deque(maxlen=buffer_size)
            self.rewards = deque(maxlen=buffer_size)
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            super(ActorCritic, self).eval()
            torch.no_grad()

        self.load()

    def calc_input_size(self):
        m = self.conv1(torch.zeros((1,) + self.obs_shape))
        #         print(m.shape)
        m = self.bn1(m)
        #         print(m.shape)
        m = self.conv2(m)
        #         print(m.shape)
        m = self.maxpool1(m)
        #         print(m.shape)
        return int(np.prod(m.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x.view((1,) + self.obs_shape)))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool1(x)
        x = F.relu(self.fc1(x.reshape(-1, 12)))
        action_prob = F.softmax(self.actor(x), dim=-1)
        if self._training:
            state_values = self.critic(x)
            return action_prob, state_values
        return action_prob, None

    def select_action(self, state, mask):
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(state)
        mask = torch.from_numpy(mask)
        m = Categorical(probs * mask)
        action = m.sample()

        if self._training:
            self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()
        # In this function, we decide whehter we want the block to move left or right,based on what the model decided

    def finish_episode(self):
        if not self._training:
            print('Training mode is disabled')
            return
        # We calculate the losses and perform backprop in this function
        R = 0
        saved_actions = [x for x in self.saved_actions]
        #     log_prob = torch.tensor([x.log_prob for x in model.saved_actions])
        #     value =
        policy_losses = []
        value_losses = []
        returns = []
        rewards = [x for x in self.rewards]

        for r in rewards[::-1]:
            R = r + 0.99 * R  # 0.99 is our gamma number
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.float().backward()
        self.optimizer.step()

        #     del model.rewards[:]
        #     del model.saved_actions[:]
        self.rewards.clear()
        self.saved_actions.clear()
        self.save()
        print('trained and saved')

    def save(self, suff=''):
        if not self._training:
            print('Training mode is disabled. Nothing to save')
            return
        if len(suff) > 0:
            suff = '_' + suff
        torch.save(self.state_dict(), model_path)

    def load(self, suff=''):
        if len(suff) > 0:
            suff = '_' + suff
        self.load_state_dict(torch.load(path.abspath(model_path)))

    def game_done(self, reward):
        self.games_played += 1
        self.wins += 1 if reward > 0 else 0
        self.history.append(reward)


#____________________________________ for example
def train(episodes_max, t_max=1000):
    #     print('target reward:', env.spec.reward_threshold)
    running_reward = 0
    for i_episode in range(episodes_max):  # We need around this much episodes
        env.reset()
        ep_reward = 0
        reward = [0, 0]
        done = [0, 0]
        for t in range(t_max):
            state, reward[t % 2], done[t % 2], _ = env.last()
            reward[t % 2] = float(reward[t % 2])
            players[t % 2].rewards.append(reward[t % 2])
            if done[t % 2]:
                if all(done):
                    players[t % 2].game_done(reward[t % 2])
                    players[(t + 1) % 2].game_done(reward[(t + 1) % 2])
                    break
                env.step(None)
                continue
            action = players[t % 2].select_action(state['observation'], state['action_mask'])
            #             ep_reward += reward
            env.step(action)

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        model_1.finish_episode()
        model_2.finish_episode()

        print("\rEpisode {}\tmodel_1 wins: {:.2f}\tmodel_2 wins: {:.2f}".format(
            i_episode, model_1.wins, model_2.wins
        ), end=' ' * 10)
        if i_episode % 100 == 0:  # We will print some things out
            print("\rEpisode {}\tmodel_1 wins: {:.2f}\tmodel_2 wins: {:.2f}".format(
                i_episode, model_1.wins, model_2.wins
            ), end=' ' * 10)
            print()
            model_1.save('last')
            model_2.save('last')

def iniate_models():
    saved_model = False
    buffer_size = 50000


    model_1 = ActorCritic('model1', obs[::-1], env.action_spaces['player_0'].n, buffer_size, lr=1e-3)
    model_2 = ActorCritic('model2', obs[::-1], env.action_spaces['player_0'].n, buffer_size, lr=5e-3)
    players = [model_1, model_2]

    if saved_model:
        model_1.load('last')
        model_2.load('last')

    eps = np.finfo(np.float32).eps.item()


def play_models():
    pl1_wins = 0
    pl2_wins = 0


    with torch.no_grad():
        for _ in range(5):
            env.reset()
            reward = [0,0]
            done = [0,0]
            for t in range(100):
                state, reward[t%2], done[t%2], _ = env.last()
                env.render()
                if done[t%2]:
                    if all(done):
                        locals()[f'pl{t%2+1}_wins'] += 1 if reward[t%2] > 0 else 0
                        locals()[f'pl{(t+1)%2+1}_wins'] += 1 if reward[(t+1)%2] > 0 else 0
                        break
                    env.step(None)
                    continue
                action = players[t%2].select_action(state['observation'].flatten(), state['action_mask'])
                env.step(action)
                sleep(0.1)
        env.close()
    print(f'{pl1_wins = }\t{pl2_wins = }')
# ________________________________  for example
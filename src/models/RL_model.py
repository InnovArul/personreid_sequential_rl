import sys, os
import numpy as np, matplotlib.pyplot as plt
from datetime import datetime
import torch, torch.optim as optim, torch.nn as nn
import random, collections, utils, copy
from collections import deque
from tqdm import tqdm
from aenum import Enum, MultiValue
from utils import get_features
from eval_metrics import evaluate

MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

LAMBDA = 0.0001    # speed of decay
MAX_EPSILON = 1
MIN_EPSILON = 0.01 # stay a bit curious even when getting old
GAMMA = 0.99 # discount factor

class Decision(Enum):
    _init_ = 'value fullname'
    _settings_ = MultiValue

    SAME = 0, 'SAME'
    DIFFERENT = 1, 'DIFFERENT'
    UNSURE = 2, 'UNSURE'

    def __int__(self):
        return self.value

class Environment:
    '''
    A gym-like environment for handling person-reid sequential multishot decision  making
    '''
    def __init__(self, person1, person2, rp, tmax = 8):
        # print('tmax', tmax)
        self.person1 = person1
        self.person2 = person2
        # features of size #persons x #frames x #featdim
        self.person1_features = self.person1['features']
        self.person2_features = self.person2['features']
        self.person1_id = self.person1['id']
        self.person2_id = self.person2['id']
        self.current_index = 0
        self.rp = rp
        
        # mandatory assertions 
        assert len(self.person1_features) >= 1, 'length of person1 features is <= 1'
        assert len(self.person2_features) >= 1, 'length of person2 features is <= 1'
        assert self.person1_features.shape[1] == self.person2_features.shape[1], 'features are not of same dimension'

        # features are expected to be of shape #frames x #feat_dim
        self.feat_dim = self.person1_features.shape[1]
        self.N = min(tmax, len(self.person1_features), len(self.person2_features))
        self.abs_features = self.get_abs_features(self.person1_features, self.person2_features, self.N)
        self.abs_features_norm = self.get_norm_of_features(self.abs_features)
        self.values_history = []

        # print(self.N, self.feat_dim, self.abs_features.shape, len(self.abs_features_norm), self.current_index, self.person1_id, self.person2_id)
    
    def get_norm_of_features(self, abs_features):
        '''to get the norm of features
        
        Arguments:
            abs_features {array float} -- list corresponding to features
        
        Returns:
            [float array] -- list of norm corresponding to features
        '''

        features_norm = []
        for i in range(len(abs_features)):
            features_norm.append(abs_features[i]. norm(p=2))
        
        features_norm = torch.stack(features_norm)
        return features_norm

    def get_abs_features(self, person1_features, person2_features, N):
        '''API to get absolute differences of features
        
        Arguments:
            person1_features {features tensor} -- list of features for person1
            person2_features {features tensor} -- list of features for person2
            N {int} -- length of this episode
        
        Returns:
            list -- list of absolute difference features 
        '''

        abs_features = []
        for i in range(N):
            abs_features.append((person1_features[i] - person2_features[i]).abs())
        
        abs_features = torch.stack(abs_features)
        return abs_features

    def get_weighted_history(self, features, values_history):
        '''to get weighted history of features in this episode so far
        
        Arguments:
            features {tensor} -- features so far
            values_history {tensor} -- values so far in the episode
        
        Returns:
            tensor -- weighted history of features
        '''
        # print(values_history[:, int(Decision.UNSURE)].unsqueeze(1).shape)
        # print(values_history.exp().sum(dim=1, keepdim=True))
        weights = 1 - (values_history[:, int(Decision.UNSURE)].unsqueeze(1).exp() / 
                       values_history.exp().sum(dim=1, keepdim=True))

        # print('weighted history', weights.shape, features.shape, values_history.shape)
        # print(weights)

        weighted_history = (features * weights).sum(dim=0)
        # print('weighted history', weights.shape, features.shape, values_history.shape, weighted_history.shape)

        return weighted_history.squeeze()

    def step(self, action, values):
        '''API to take an action
        
        Arguments:
            action {int} -- action to be taken
            values {tensor} -- values given by Q-learning model
        
        Returns:
            next_state, reward, done, info -- similar to open AI gym
        '''

        # print(values.shape)
        assert ((action == int(Decision.SAME)) or
                (action == int(Decision.DIFFERENT)) or
                (action == int(Decision.UNSURE))), 'action is not valid'

        # append the values to history
        self.values_history.append(values)
        next_state, reward, done, info = None, None, None, None
        
        if action == int(Decision.SAME):
            # if action is SAME and if the ID's are same, reward +1
            if self.person1_id == self.person2_id:
                reward = +1
            else:
                reward = -1
            
            done = True
            next_state = None

        elif action == int(Decision.DIFFERENT):
            # if action is DIFFERENT and if the ID's are not same, reward +1
            if self.person1_id != self.person2_id:
                reward = +1
            else:
                reward = -1
            
            done = True
            next_state = None
        
        else:
            # if action is UNSURE and if there features are depleted, reward -1
            if self.current_index >= self.N:
                reward = -1
                done = True
                next_state = None
            else:
                # if there are more frames in the episode, determine the content for next_state
                next_state_feature = self.abs_features[self.current_index]
                next_state_history = self.get_weighted_history(self.abs_features[:self.current_index], 
                                                               torch.cat(self.values_history))
                next_state_norm = self.abs_features_norm[:self.current_index+1]
                next_state = self.create_state(next_state_feature, next_state_history, next_state_norm)
                self.current_index += 1

                done = False
                reward = self.rp

        return next_state, reward, done, info

    def create_state(self, feature, history, features_norm):
        # print(feature.shape, history.shape)
        current_observation = torch.cat([feature, history, features_norm.min().view(-1), 
                                                  features_norm.max().view(-1), features_norm.mean().view(-1)])
        return current_observation

    def reset(self):
        # give out the 0th frame features
        self.values_history = []
        self.current_index = 1
        current_observation = self.create_state(self.abs_features[0], self.abs_features[0], self.abs_features_norm[0])
        # print(current_observation.shape)
        return current_observation


# experience replay buffer
class Memory:
    def __init__(self, size):
        self.size = size
        self.samples = deque()
    
    def append(self, x):
        # reduce the size until it become self.size 
        if isinstance(x, collections.Iterable):
            # if it is array, the add it
            in_items_len = len(x)
            while (len(x) + in_items_len) >= self.size:
                x.popleft()
            
            self.samples += x
        else:
            # if it is single element, append it
            while (len(x) + 1) >= self.size:
                x.popleft()
            
            self.samples.append(x)
    
    def sample(self, n):
        # sample random n samples
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

class Brain(nn.Module):
    """Core logic of DQN
    """

    def __init__(self, nStateDim, nActions):
        super().__init__()

        # an MLP for state-action value function
        self.state_action_value = nn.Sequential(
            nn.Linear(nStateDim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),            
            nn.Linear(128, nActions)
        )

    def forward(self, x):
        # if np array, convert into torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().cuda()

        # if state has no batch size, create a dummy batch with one sample
        if x.ndimension() == 1:
            x = x.unsqueeze(0)

        return self.state_action_value(x)
    
class Agent(nn.Module):
    def __init__(self, feature_extractor, args, nBufferSize=MEMORY_CAPACITY):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # input is different features, history, 3 handcrafted features
        self.brain = Brain(feature_extractor.feat_dim * 2 + 3, 3)
        if not args.use_cpu:
            self.brain = self.brain.cuda()

        self.memory = Memory(nBufferSize)
        self.rp = args.rp

        # optimizer setup
        self.steps = 0
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def act(self, s, epsilon):
        '''epsilon greedy action selection
        
        Arguments:
            s {np array} -- state selection
            epsilon {float} -- epsilon value
        
        Returns:
            action -- action index
        '''

        q_values = self.brain(s)
        if epsilon > np.random.uniform():
            action = np.random.choice(3)
        else:  
            with torch.no_grad():
                action = torch.argmax(q_values, dim=1).item()

        return action, q_values
    
    def get_epsilon(self, epoch):
        '''get epsilon value
        
        Returns:
            float -- epsilon value
        '''

        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)
        # 1 - (epoch - 1) * 0.1
        # self.epsilon = max(0.05, self.epsilon)
        # MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)
        self.steps += 1
        return self.epsilon

    def play_one_episode(self, epoch=0, is_test=False):
        '''play one episode of cart pole
        
        Keyword Arguments:
            is_test {bool} -- is it a test episode? (default: {False})
        
        Returns:
            samples, total_reward, iters -- collected samples, rewards and total iterations
        '''

        current_state = self.env.reset()
        total_reward = 0
        iters = 0
        samples = []
        done = False
        # print('start')

        while not done:
            # sample an action according to epsilon greedy strategy
            epsilon = 0
            if not is_test:
                epsilon = self.get_epsilon(epoch)
            else:
                epsilon = 0
            
            # print(current_state.shape)
            action, q_values = self.act(current_state, epsilon)
            # print(q_values.shape)
            next_state, reward, done, info = self.env.step(action, q_values)
            # print('next', action, reward, done)
            # input()
            iters += 1
            total_reward += reward
            samples.append([current_state, action, reward, next_state])
            current_state = next_state

        return samples, total_reward, iters, q_values

    def collect_data(self, dataloader, epoch, args):
        '''collect data and put it into replay memory
        
        Arguments:
            datacol_num_episodes {int} -- number of episodes to run and collect data
        
        Returns:
            avg_reward -- average reward in data collection
        '''

        total_datacol_rewards = 0
        total_train_loss = 0
        self.feature_extractor.eval()

        if not args.use_cpu:
            self.feature_extractor = self.feature_extractor.cuda()

        for i, data in enumerate(tqdm(dataloader)):
            person1_frames, person1_id, person2_frames, person2_id = data
            if not args.use_cpu:
                person1_frames = person1_frames.cuda()
                person2_frames = person2_frames.cuda()

            with torch.no_grad():
                _, person1_features = self.feature_extractor(person1_frames)
                _, person2_features = self.feature_extractor(person2_frames)

            self.env = Environment({'features': person1_features.squeeze(), 'id':person1_id}, 
                                   {'features': person2_features.squeeze(), 'id':person2_id}, self.rp)

            # play each episode and put the samples into memory
            samples, total_reward, iters, _ = self.play_one_episode(epoch=epoch)
            self.memory.append(samples)
            total_datacol_rewards += total_reward
        
        avg_reward = total_datacol_rewards / len(dataloader)
        return avg_reward

    def get_target_Q_values(self, rewards, next_states, model):
        '''get the target Q values for Q-learning
        
        Arguments:
            rewards {tensor} -- reward for each transition
            next_states {array of list} -- state description
            model {DQ network} -- target model
        
        Returns:
            Q target -- Q target values
        '''

        targetQvals = torch.zeros(len(next_states), 1).to('cuda')

        for i, next_state in enumerate(next_states):
            if next_state is None:
                targetQvals[i, 0] = rewards[i]
            else:
                state_torch = next_state.unsqueeze(0).cuda()
                with torch.no_grad():
                    q_values = model(state_torch)
                
                max_q_value = torch.max(q_values, dim=1)[0].item()
                targetQvals[i, 0] = rewards[i] + GAMMA * max_q_value

        return targetQvals
    
    def tensorize_samples(self, samples):
        '''collect the current states, rewards, actions, next states from the sampled data from 
        experience replay buffer
        
        Arguments:
            samples {array of list} -- data sampled from replay buffer
        
        Returns:
            current_states, actions, rewards, next_states -- separated data
        '''

        current_states, actions, rewards, next_states = [],[],[],[]
        for _, data in enumerate(samples):
            current_states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])

        current_states = torch.stack(current_states)
        actions = torch.from_numpy(np.array(actions)).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float)).float().unsqueeze(1)
        return current_states, actions, rewards, next_states

    def train(self, args, num_runs = 100):
        '''train the DQN for num_runs iterations
        
        Keyword Arguments:
            num_runs {int} -- number of iterations to train (default: {100})
        
        Returns:
            avg_train_loss -- average train loss
        '''

        # backup the brain for fixed target
        target_model = copy.deepcopy(self.brain)
        total_loss = 0
        self.brain.train()
        self.feature_extractor.eval()
        if not args.use_cpu:
            self.brain = self.brain.cuda()

        for i in tqdm(range(num_runs)):
            # sample data from memory
            data = self.memory.sample(BATCH_SIZE)
            current_states, actions, rewards, next_states = self.tensorize_samples(data)

            if not args.use_cpu:
                current_states = current_states.cuda().detach().clone()
                actions = actions.cuda().detach().clone()
                rewards = rewards.cuda().detach().clone()

            # get max Q values of next state to form fixed target
            target_Q = self.get_target_Q_values(rewards, next_states, target_model)

            # update the model
            currentQ = self.brain(current_states)
            self.optimizer.zero_grad()
            loss = torch.mean((target_Q - currentQ.gather(1, actions))**2)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
    
        return total_loss / num_runs
    
    def select_random_features(self, source_features, num_features):
        # source features is of size #len x #feat_dim
        # we have to select random #num_features
        num_features = min(num_features, source_features.shape[0])
        indices = np.random.choice(range(num_features), size=num_features, replace=False)
        return source_features[indices]

    def test(self, queryloader, galleryloader, args, ranks=[1, 5, 10, 20]):
        self.brain.eval()
        self.feature_extractor.eval()

        buffer_file = '../scratch/' + args.dataset + '_test_features.pth'
        if not os.path.exists(buffer_file):
            # if the buffer files saved with features already existing, load buffer file
            # if not, extract the features from feature extractor
            qindividualf, qmeanf, q_pids, q_camids = [], [], [], []
            print('extracting query feats')
            for batch_idx, (imgs, pids, camids) in enumerate(tqdm(queryloader)):
                if not args.use_cpu:
                    imgs = imgs.cuda()

                with torch.no_grad():
                    # b=1, n=number of clips, s=16
                    b, n, s, c, h, w = imgs.size()
                    assert(b==1)
                    imgs = imgs.view(b*n, s, c, h, w)
                    individual_features = get_features(self.feature_extractor, imgs, args.test_num_tracks)
                    mean_features = torch.mean(individual_features, 0)
                    
                    individual_features = individual_features.data.cpu()
                    mean_features = mean_features.data.cpu()

                    qindividualf.append(individual_features)
                    qmeanf.append(mean_features)
                    q_pids.extend(pids)
                    q_camids.extend(camids)
                    torch.cuda.empty_cache()

            qmeanf = torch.stack(qmeanf)
            q_pids = np.asarray(q_pids)
            q_camids = np.asarray(q_camids)

            print("Extracted features for query set, obtained {}-by-{} matrix".format(qmeanf.size(0), qmeanf.size(1)))

            gindividualf, gmeanf, g_pids, g_camids = [], [], [], []
            print('extracting gallery feats')
            for batch_idx, (imgs, pids, camids) in enumerate(tqdm(galleryloader)):
                if not args.use_cpu:
                    imgs = imgs.cuda()

                with torch.no_grad():
                    b, n, s, c, h, w = imgs.size()
                    imgs = imgs.view(b*n, s , c, h, w)
                    assert(b==1)
                    # handle chunked data
                    individual_features = get_features(self.feature_extractor, imgs, args.test_num_tracks)
                    mean_features = torch.mean(individual_features, 0)
                    torch.cuda.empty_cache()

                individual_features = individual_features.data.cpu()
                mean_features = mean_features.data.cpu()

                gindividualf.append(individual_features)
                gmeanf.append(mean_features)
                g_pids.extend(pids)
                g_camids.extend(camids)

            gmeanf = torch.stack(gmeanf)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)

            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gmeanf.size(0), gmeanf.size(1)))
            torch.save({'query':{'meanft':qmeanf,
                                 'individualft':qindividualf,
                                 'pids':q_pids,
                                 'camids':q_camids},
                        'gallery':{'meanft':gmeanf,
                                 'individualft':gindividualf,
                                 'pids':g_pids,
                                 'camids':g_camids}
                        }, buffer_file)
        
        else:
            # load the buffer file
            print('loading and extraction information/features from file', buffer_file)
            buffer = torch.load(buffer_file)
            qmeanf = buffer['query']['meanft']
            qindividualf = buffer['query']['individualft']
            q_camids = buffer['query']['camids']
            q_pids = buffer['query']['pids']
            gmeanf = buffer['gallery']['meanft']
            gindividualf = buffer['gallery']['individualft']
            g_camids = buffer['gallery']['camids']
            g_pids = buffer['gallery']['pids']            

        print("Computing distance matrix for allframes evaluation (baseline)")
        m, n = qmeanf.size(0), gmeanf.size(0)
        distmat = torch.pow(qmeanf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gmeanf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qmeanf, gmeanf.t())
        distmat = distmat.numpy()
        print("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
        print("------------------")

        print('Computing distance matrix from DQN network')
        distmat = torch.zeros(m, n)
        instance_rewards = torch.zeros(m, n)
        comparisons = torch.zeros(m, n)
        print(m, n)
        for i, qid in tqdm(enumerate(q_pids)):
            q_features = self.select_random_features(qindividualf[i], args.rl_seq_len)
            for j, gid in enumerate(g_pids):
                # print(qindividualf[i].shape, gindividualf[j].shape)
                g_features = self.select_random_features(gindividualf[j], args.rl_seq_len)

                if not args.use_cpu:
                    q_features = q_features.cuda()
                    g_features = g_features.cuda()

                # print(q_features.shape, g_features.shape)
                self.env = Environment({'features':q_features,
                                        'id':qid},
                                       {'features':g_features,
                                        'id':gid}, args.rp)
                _, reward, iters, q_vals = self.play_one_episode(is_test=True)
                instance_rewards[i,j] = reward
                comparisons[i,j] = iters
                distmat[i,j] = (q_vals[:,1] - q_vals[:,0]).item()
                # break

        print("Computing CMC and mAP (+ve distmat)")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
        print("------------------")


        print("Computing CMC and mAP (-ve distmat)")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
        print("------------------")

        print('average rewards', instance_rewards.mean().item())
        print('average comparison', comparisons.mean().item())

        return cmc[0]

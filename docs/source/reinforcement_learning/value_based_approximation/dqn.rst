====================
Deep Q-Network (DQN)
====================

Atari Wrappers
==============

.. code:: python

    class NoopResetEnv(gym.Wrapper):
        def __init__(self, env, noop_max=30):
            """Sample initial states by taking random number of no-ops on reset.
            No-op is assumed to be action 0.
            """
            gym.Wrapper.__init__(self, env)
            self.noop_max = noop_max
            self.noop_action = 0
            assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

        def reset(self, **kwargs):
            """ Do no-op action for a number of steps in [1, noop_max]."""
            self.env.reset(**kwargs)
            noops = np.random.randint(1, self.noop_max + 1)
            assert noops > 0
            obs = None
            for _ in range(noops):
                obs, _, done, _ = self.env.step(self.noop_action)
                if done:
                    obs = self.env.reset(**kwargs)
            return obs

        def step(self, action):
            return self.env.step(action)


.. code:: python

    class FireResetEnv(gym.Wrapper):
        def __init__(self, env):
            """Take action on reset for environments that are fixed until firing."""
            gym.Wrapper.__init__(self, env)
            assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
            assert len(env.unwrapped.get_action_meanings()) >= 3

        def reset(self, **kwargs):
            self.env.reset(**kwargs)
            obs, _, done, _ = self.env.step(1)
            if done:
                self.env.reset(**kwargs)
            obs, _, done, _ = self.env.step(2)
            if done:
                self.env.reset(**kwargs)
            return obs

        def step(self, action):
            return self.env.step(action)

.. code:: python

    class EpisodicLifeEnv(gym.Wrapper):
        def __init__(self, env):
            """Make end-of-life == end-of-episode, but only reset on true game over.
            Done by DeepMind for the DQN and co. since it helps value estimation.
            """
            gym.Wrapper.__init__(self, env)
            self.lives = 0
            self.was_real_done  = True

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.was_real_done = done
            # check current lives, make loss of life terminal,
            # then update lives to handle bonus lives
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                # for Qbert sometimes we stay in lives == 0 condition for a few frames
                # so it's important to keep lives > 0, so that we only reset once
                # the environment advertises done.
                done = True
            self.lives = lives
            return obs, reward, done, info
        
        
        def reset(self, **kwargs):
            """Reset only when lives are exhausted.
            This way all states are still reachable even though lives are episodic,
            and the learner need not know about any of this behind-the-scenes.
            """
            if self.was_real_done:
                obs = self.env.reset(**kwargs)
            else:
                # no-op step to advance from terminal/lost life state
                obs, _, _, _ = self.env.step(0)
            self.lives = self.env.unwrapped.ale.lives()
            return obs


.. code:: python

    class MaxAndSkipEnv(gym.Wrapper):
        def __init__(self, env, skip=4):
            """Return only every `skip`-th frame"""
            gym.Wrapper.__init__(self, env)
            # most recent raw observations (for max pooling across time steps)
            self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
            self._skip       = skip

        def step(self, action):
            """Repeat action, sum reward, and max over last observations."""
            total_reward = 0.0
            done = None
            for i in range(self._skip):
                obs, reward, done, info = self.env.step(action)
                if i == self._skip - 2: self._obs_buffer[0] = obs
                if i == self._skip - 1: self._obs_buffer[1] = obs
                total_reward += reward
                if done:
                    break
            # Note that the observation on the done=True frame
            # doesn't matter
            max_frame = self._obs_buffer.max(axis=0)

            return max_frame, total_reward, done, info

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)


.. code:: python

    class ClipRewardEnv(gym.RewardWrapper):
        def __init__(self, env):
            gym.RewardWrapper.__init__(self, env)

        def reward(self, reward):
            """Bin reward to {+1, 0, -1} by its sign."""
            return np.sign(reward)


.. code:: python

    class WarpFrame(gym.ObservationWrapper):
        def __init__(self, env, width=84, height=84):
            """
            Warp frames to 84x84 as done in the Nature paper and later work.
            """
            super().__init__(env)
            self._width = width
            self._height = height

            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(1, self._height, self._width),
                dtype=np.float32,
            )


        def observation(self, obs):

            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(
                obs, (self._width, self._height), interpolation=cv2.INTER_AREA
            )

            obs = np.expand_dims(obs, 0)
            return obs


.. code:: python

    class FrameStack(gym.Wrapper):
        def __init__(self, env, k):
            """Stack k last frames"""
            gym.Wrapper.__init__(self, env)
            self.k = k
            self.frames = deque([], maxlen=k)
            shp = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=((k,)+shp[1:]), dtype=env.observation_space.dtype)

        def reset(self):
            obs = self.env.reset()
            for _ in range(self.k):
                self.frames.append(obs)
            
            return np.array(self.frames).reshape(self.observation_space.shape)
            
        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.frames.append(obs)
            obs = np.array(self.frames).reshape(self.observation_space.shape)
            return obs, reward, done, info

.. code:: python

    def create_atari_env(name):
        env = gym.make(name)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, 4)
        return env


Experience Replay
=================

.. code:: python

    class MemoryBuffer:
    
        def __init__(self, obs_shape, max_len, batch_size):
            self.idx = 0
            self.max_len = max_len
            self.current_len = 0
            self.batch_size = batch_size
            
            self.obs = np.zeros(shape=(max_len, *obs_shape), dtype=np.float32)
            self.action = np.zeros(shape=(max_len, 1), dtype=np.float32)
            self.reward = np.zeros(shape=(max_len, 1), dtype=np.float32)
            self.next_obs = np.zeros(shape=(max_len, *obs_shape), dtype=np.float32)
            self.done  = np.zeros(shape=(max_len, 1), dtype=np.float32)
            
        def __len__(self):
            return self.current_len
        
        def add_experience(self, obs, action, reward, next_obs, done):
            self.obs[self.idx] = obs
            self.action[self.idx] = action
            self.reward[self.idx] = reward
            self.next_obs[self.idx] = next_obs
            self.done[self.idx] = done
            
            self.idx = (self.idx + 1) % self.max_len
            self.current_len = min(self.current_len + 1, self.max_len)
        
        def draw_samples(self):
            
            idxs = np.random.choice(len(self), self.batch_size, replace=False)
            
            obs = self.obs[idxs]
            action = self.action[idxs]
            reward = self.reward[idxs]
            next_obs = self.next_obs[idxs]
            done = self.done[idxs]
            
            return obs, action, reward, next_obs, done


Action-Value Function
=====================

.. code:: python

    class Q(nn.Module):
        
        def __init__(self, n_actions):
            super(Q, self).__init__()
            
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=64*7*7, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=n_actions)
            )

        def forward(self, state):
            return self.model(state)


Agent
=====

.. code:: python

    class Agent:
    
        def __init__(self,
                    obs_shape,
                    n_actions,
                    batch_size, 
                    memory_size,
                    update_frequency,
                    warmup,
                    alpha, 
                    epsilon_start, 
                    epsilon_steps, 
                    epsilon_end, 
                    gamma):
            
            self.n_actions = n_actions
            self.memory_buffer = MemoryBuffer(obs_shape, memory_size, batch_size)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(self.device)
            
            self.online_network = Q(n_actions).to(self.device)
            self.target_network = deepcopy(self.online_network).to(self.device)
            
            for param in self.target_network.parameters():
                param.requires_grad = False
            
            self.optimizer = optim.RMSprop(self.online_network.parameters(), alpha)
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_step = (epsilon_start - epsilon_end) / epsilon_steps
            print(self.epsilon_step)
            self.gamma = gamma
            self.warmup = warmup
        
        
        def adjust_epsilon(self):
            self.epsilon -= self.epsilon_step
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end
        
        @torch.no_grad()
        def epsilon_greedy(self, obs):
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.n_actions)
            else:
                action = self.greedy(obs)
            return action
        
        @torch.no_grad()
        def greedy(self, obs):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            return self.online_network(obs).argmax().item()
        
        def store_memory(self, obs, action, reward, next_obs, done):
            self.memory_buffer.add_experience(obs, action, reward, next_obs, done)
        
        def batch_memory(self):
            obs, action, reward, next_obs, done = self.memory_buffer.draw_samples()
            
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.device)
                    
            return obs, action, reward, next_obs, done
            
        def learn(self):
            if len(self.memory_buffer) < self.warmup:
                return
            
            self.optimizer.zero_grad()
            obs, action, reward, next_obs, done = self.batch_memory()
            
            with torch.no_grad():
                target = reward + self.gamma * self.target_network(next_obs).max(dim=1, keepdim=True)[0] * (1 - done)

            
            online = self.online_network(obs).gather(dim=1, index=action)
                    
            td_error = target - online
            loss = td_error.pow(2).mul(0.5).mean()
            loss.backward()
            self.optimizer.step()
            
            self.adjust_epsilon()
            
        def update_target_network(self):
            self.target_network = deepcopy(self.online_network)


Main Training Loop
==================

.. code:: python

    # parameters
    env_name = 'BreakoutNoFrameskip-v4'

    EPISODES = 100000
    BATCH_SIZE = 32
    MEMORY_SIZE = 100000
    UPDATE_FREQUENCY = 10000
    WARMUP = 1000
    ALPHA = 0.00025
    EPSILON_START = 1
    EPSILON_END = 0.1
    EPSILON_STEPS = 100000
    GAMMA = 0.99

    # training loop
    def main():
        env = create_atari_env(env_name)
        agent = Agent(
            env.observation_space.shape,
            env.action_space.n,
            BATCH_SIZE,
            MEMORY_SIZE,
            UPDATE_FREQUENCY,
            WARMUP,
            ALPHA,
            EPSILON_START,
            EPSILON_STEPS,
            EPSILON_END,
            GAMMA
        )
        counter = 1
        for episode in range(EPISODES):
            obs = env.reset()
            done = False
            
            reward_sum = 0
            while not done:
                counter+=1
                action = agent.epsilon_greedy(obs)
                next_obs, reward, done, info = env.step(action)
                agent.store_memory(obs, action, reward, next_obs, done)
                obs = next_obs
                agent.learn()
                
                if counter % UPDATE_FREQUENCY == 0:
                    print("Updating")
                    agent.update_target_network()
                    
                reward_sum += reward
            
            print(f'Episode: {episode}, Counter: {counter}, Epsilon: {agent.epsilon}, Reward: {reward_sum}')
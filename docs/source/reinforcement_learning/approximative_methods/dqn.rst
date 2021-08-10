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
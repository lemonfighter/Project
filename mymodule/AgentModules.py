import tensorflow.compat.v1 as tf
import numpy as np


def load_model(path):
    return tf.keras.models.load_model(path)


class ValueBase:
    def __init__(self,
                 model: tf.keras.models.Model,
                 action_dim: int,
                 gamma=0.99,
                 replace_point=1024,
                 memory_size=4096,
                 memory_batch=64,
                 epsilon=1,
                 min_epsilon=0.001,
                 decay_epsilon=0.995,
                 double=True):
        # Model
        self.eval_model = model
        self.target_model = model
        self.action_dim = action_dim
        self.double = double

        # Setting
        self.gamma = gamma
        self.replace_point = replace_point
        self.replace_pointer = 0

        # Memory
        self.memory = None
        self.memory_batch = memory_batch
        self.memory_size = memory_size
        self.memory_pointer = 0

        # Greedy
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_epsilon = decay_epsilon

    def memory_storage(self, state, action, reward, next_state, done):
        if not isinstance(state, list):
            state = [state]
            next_state = [next_state]
        if self.memory is None:
            self.memory = {
                's': [np.zeros((self.memory_size,) + ss.shape[1:]) for ss in state],
                'a': np.zeros((self.memory_size, 1), dtype=int),
                'r': np.zeros((self.memory_size, 1), dtype=float),
                's_': [np.zeros((self.memory_size,) + ss.shape[1:]) for ss in next_state],
                'd': np.zeros((self.memory_size, 1), dtype=int),
            }
        i = self.memory_pointer % self.memory_size
        for ii, ss in enumerate(state):
            self.memory['s'][ii][i] = ss

        if done:
            for ii, ss in enumerate(state):
                self.memory['s_'][ii][i] = np.zeros_like(self.memory['s_'][ii][0])
        else:
            for ii, ss in enumerate(next_state):
                self.memory['s_'][ii][i] = ss
        self.memory['a'][i] = action
        self.memory['r'][i] = reward
        self.memory['d'][i] = done
        self.memory_pointer += 1

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        act_values = self.eval_model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self) -> float:
        i = np.random.choice(self.memory_size, size=self.memory_batch, replace=False)
        bs = [ss[i] for ss in self.memory['s']]
        ba = self.memory['a'][i]
        br = self.memory['r'][i]
        bs_ = [ss[i] for ss in self.memory['s_']]
        bd = self.memory['d'][i]

        eval_y = self.eval_model.predict(bs)
        target_y = self.target_model.predict(bs_)

        if self.double:
            eval_a = np.argmax(self.eval_model.predict(bs_), axis=1)
            ys = br + self.gamma * target_y[np.arange(self.memory_batch), eval_a].reshape(-1, 1) * (1 - bd)
        else:
            ys = br + self.gamma * np.max(target_y, axis=1).reshape(-1, 1) * (1 - bd)

        for i, a, y in zip(range(self.memory_batch), ba, ys):
            eval_y[i, a] = y

        loss = self.eval_model.train_on_batch(bs, eval_y)

        if self.replace_pointer % self.replace_point == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
        self.replace_pointer += 1

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_epsilon
        return loss

    def save(self, path):
        self.eval_model.save(path)

    def load(self, path):
        self.eval_model = tf.keras.models.load_model(path)
        self.target_model = self.eval_model

    # Example 1
    '''
    import gym

    env = gym.make('CartPole-v1')


    def build_model():
        layer = tf.keras.layers
        model = tf.keras.models.Sequential([
            layer.Dense(32, input_shape=(4,), activation=tf.keras.activations.relu),
            layer.Dense(16, activation=tf.keras.activations.relu),
            layer.Dense(2, activation=tf.keras.activations.linear)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss=tf.keras.losses.mean_squared_error,
                      metrics=['accuracy'])
        return model


    print('Input Dim: {}'.format(env.observation_space))
    print('Action Dim: {}'.format(env.action_space.n))
    env = env.unwrapped

    agent = ValueBase(build_model(),
                      action_dim=env.action_space.n,
                      gamma=0.95,
                      replace_point=512,
                      memory_size=4096,
                      memory_batch=64)

    for cot in range(500):
        ep_r = 0
        s = env.reset().reshape(1, -1)
        d = False
        step = 0
        while step < 200:
            a = agent.choose_action(s)
            s_, r, d, _ = env.step(a)
            r = r if not d else -10
            s_ = s_.reshape(1, -1)

            agent.memory_storage(s, a, r, s_, d)
            s = s_
            ep_r += r
            step += 1
            if d:
                break
            if agent.memory_pointer > agent.memory_size:  # learning
                env.render()
                agent.train()
        if agent.memory_pointer > agent.memory_size:
            print('Round : {}, R : {}'.format(cot + 1, ep_r))

    '''
    # Example 2
    '''
    import gym

    env = gym.make('Pendulum-v0')


    def build_model():
        layer = tf.keras.layers
        model = tf.keras.models.Sequential([
            layer.Dense(32, input_shape=(3,), activation=tf.keras.activations.relu),
            layer.Dense(32, activation=tf.keras.activations.relu),
            layer.Dense(11, activation=tf.keras.activations.linear)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss=tf.keras.losses.mean_squared_error,
                      metrics=['accuracy'])
        return model


    print('Input Dim: {}'.format(env.observation_space))
    print('Action Dim: {}'.format(env.action_space))
    env = env.unwrapped

    agent = ValuesBase(build_model(),
                       action_dim=3,
                       gamma=0.95,
                       replace_point=512,
                       memory_size=8192,
                       memory_batch=128,)
    anp = np.linspace(-2, 2, 11)
    for cot in range(500):
        ep_r = 0
        s = env.reset().reshape(1, -1)
        d = False
        step = 0
        while step < 1000:
            a = agent.choose_action(s)
            s_, r, d, _ = env.step([anp[a]])
            r = r if not d else -10
            s_ = s_.reshape(1, -1)

            agent.storage(s, a, r, s_, d)
            s = s_
            ep_r += r
            step += 1
            if d:
                break
            if agent.memory_pointer > agent.memory_size:  # learning
                env.render()
                agent.train()
        if agent.memory_pointer > agent.memory_size:
            print('Round : {}, R : {}'.format(cot + 1, ep_r))
    '''


class PolicyBase:
    def __init__(self,
                 model: tf.keras.models.Model,
                 action_dim: int,
                 discount=0.99,
                 gamma=0.001):
        # Model
        self.model_loss = model.loss
        self.model = model
        self.model.compile(optimizer=model.optimizer, loss=self._loss)
        self.action_dim = action_dim

        # Setting
        self.discount = discount
        self.gamma = gamma

        # Memory
        self.memory = None

    def _loss(self, true, pred):
        true, reward = true[:, :-1], true[:, -1]
        return self.model_loss(true, pred) * reward

    def memory_storage(self, state, action: int, reward: float):
        if not isinstance(state, list):
            state = [state]

        if self.memory is None:
            self.memory = {
                's': [ss for ss in state],
                'a': [],
                'r': [],
            }
            self.memory['a'].append(action)
            self.memory['r'].append(reward)
        else:
            for i in range(len(state)):
                self.memory['s'][i] = np.concatenate((self.memory['s'][i], state[i]))
            self.memory['a'].append(action)
            self.memory['r'].append(reward)

    def choose_action(self, state, get_action_value=False):
        av = self.model.predict(state)[0]
        if get_action_value:
            return av
        prob = av / np.sum(av)
        return int(np.random.choice(self.action_dim, p=prob))

    def train(self):
        def discount_r():
            dr = np.zeros_like(br, dtype=np.float32)
            cot_r = 0
            for i in reversed(range(len(dr))):
                cot_r = cot_r * self.discount + br[i]
                dr[i] = cot_r
            dr -= dr.mean()
            dr /= (dr.std() + 1e-20)
            return dr.reshape(-1, 1)

        bs = [ss for ss in self.memory['s']]
        ba = np.array(self.memory['a'])
        br = np.array(self.memory['r'])

        bav = self.model.predict(bs)
        discountr = discount_r()

        gradients = np.zeros_like(bav)
        gradients[np.arange(len(gradients)), ba] = 1

        self.model.train_on_batch(bs, np.concatenate((gradients, discountr), axis=1))
        self._clearn_memory()

    def _clearn_memory(self):
        self.memory = None

    # Example 1
    '''
    import gym

    env = gym.make('CartPole-v1')


    def build_model():
        layer = tf.keras.layers
        model = tf.keras.models.Sequential([
            layer.Dense(32, input_shape=(4,), activation=tf.keras.activations.relu),
            layer.Dense(16, activation=tf.keras.activations.relu),
            layer.Dense(2, activation=tf.keras.activations.softmax)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss=tf.keras.losses.categorical_crossentropy)
        return model


    print('Input Dim: {}'.format(env.observation_space))
    print('Action Dim: {}'.format(env.action_space.n))
    env = env.unwrapped

    agent = PolicyBase(build_model(),
                       action_dim=env.action_space.n,)

    for cot in range(5000):
        ep_r = 0
        s = env.reset().reshape(1, -1)
        d = False
        step = 0
        while not d:
            a = agent.choose_action(s)
            s_, r, d, _ = env.step(a)
            r = r if not d else -10
            s_ = s_.reshape(1, -1)

            agent.memory_storage(s, a, r)
            s = s_
            ep_r += r
            step += 1
            if d:
                agent.train()
        print('Round : {}, R : {}'.format(cot + 1, ep_r))
    '''


class ACModel:
    def __init__(self,
                 actor_model: tf.keras.models.Model,
                 critic_model: tf.keras.models.Model,
                 action_dim: int,
                 gamma=0.95):
        # Model
        self.actor_loss = actor_model.loss
        self.actor_model = actor_model
        self.actor_model.compile(optimizer=actor_model.optimizer, loss=self._loss)
        self.critic_model = critic_model
        self.action_dim = action_dim

        # Setting
        self.gamma = gamma

        self.s = None
        self.a_prob = None
        self.a = None

    def _loss(self, true, pred):
        true, reward = true[:, :-1], true[:, -1]
        return self.actor_loss(true, pred) * reward

    def choose_action(self, state, get_action_value=False):
        self.a_prob = self.actor_model.predict(state)[0]
        prob = self.a_prob / np.sum(self.a_prob)

        self.s = state
        self.a = int(np.random.choice(self.action_dim, p=prob))

        if get_action_value:
            return self.a_prob
        return self.a

    def train(self, state, action, reward, next_state, done):
        target_cy = reward + self.gamma * self.critic_model.predict(next_state) * (1 - done)

        td_error = target_cy - self.critic_model.predict(state)

        self.critic_model.train_on_batch(state, target_cy)

        gradient = np.zeros((1, self.action_dim))
        gradient[0, action] = 1
        self.actor_model.train_on_batch(state, np.concatenate((gradient, td_error), axis=1))
    # Example 1
    '''
    import gym

    env = gym.make('CartPole-v1')


    def build_amodel():
        layer = tf.keras.layers
        model = tf.keras.models.Sequential([
            layer.Dense(32, input_shape=(4,), activation=tf.keras.activations.relu),
            layer.Dense(16, activation=tf.keras.activations.relu),
            layer.Dense(2, activation=tf.keras.activations.softmax)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
                      loss=tf.keras.losses.categorical_crossentropy)

        return model


    def build_cmodel():
        layer = tf.keras.layers
        model = tf.keras.models.Sequential([
            layer.Dense(32, input_shape=(4,), activation=tf.keras.activations.relu),
            layer.Dense(8, activation=tf.keras.activations.relu),
            layer.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.mean_squared_error)

        return model


    print('Input Dim: {}'.format(env.observation_space))
    print('Action Dim: {}'.format(env.action_space.n))
    env = env.unwrapped

    agent = ACModel(actor_model=build_amodel(),
                    critic_model=build_cmodel(),
                    action_dim=env.action_space.n,)

    for cot in range(5000):
        ep_r = 0
        s = env.reset().reshape(1, -1)
        d = False
        step = 0
        while not d:
            # env.render()
            a = agent.choose_action(s)
            s_, r, d, _ = env.step(a)
            r = r if not d else -10
            s_ = s_.reshape(1, -1)

            agent.train(s, a, r, s_, d)
            # agent.memory_storage(s, a, r, s_, d)
            # if agent.memory_pointer > agent.memory_size:
            #     agent.train()
            s = s_
            ep_r += r
            step += 1
        print('Round : {}, R : {}'.format(cot + 1, ep_r))
    '''


class A3CModel(object):
    pass


if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v1')


    def build_amodel():
        layer = tf.keras.layers
        model = tf.keras.models.Sequential([
            layer.Dense(32, input_shape=(4,), activation=tf.keras.activations.relu),
            layer.Dense(16, activation=tf.keras.activations.relu),
            layer.Dense(2, activation=tf.keras.activations.softmax)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002),
                      loss=tf.keras.losses.categorical_crossentropy)

        return model


    def build_cmodel():
        layer = tf.keras.layers
        model = tf.keras.models.Sequential([
            layer.Dense(32, input_shape=(4,), activation=tf.keras.activations.relu),
            layer.Dense(8, activation=tf.keras.activations.relu),
            layer.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                      loss=tf.keras.losses.mean_squared_error)

        return model


    print('Input Dim: {}'.format(env.observation_space))
    print('Action Dim: {}'.format(env.action_space.n))
    env = env.unwrapped

    agent = ACModel(actor_model=build_amodel(),
                    critic_model=build_cmodel(),
                    action_dim=env.action_space.n,)

    for cot in range(5000):
        ep_r = 0
        s = env.reset().reshape(1, -1)
        d = False
        step = 0
        while not d:
            # env.render()
            a = agent.choose_action(s)
            s_, r, d, _ = env.step(a)
            r = r if not d else -10
            s_ = s_.reshape(1, -1)

            agent.train(s, a, r, s_, d)
            # agent.memory_storage(s, a, r, s_, d)
            # if agent.memory_pointer > agent.memory_size:
            #     agent.train()
            s = s_
            ep_r += r
            step += 1
        print('Round : {}, R : {}'.format(cot + 1, ep_r))

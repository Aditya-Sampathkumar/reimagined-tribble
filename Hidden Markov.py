import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions

initial_distributions = tfd.Categorical(probs=[0.8, 0.2])
transition_distributions = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observation_distributions = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(initial_distribution=initial_distributions, transition_distribution=transition_distributions, observation_distribution=observation_distributions, num_steps=7)

mean = model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
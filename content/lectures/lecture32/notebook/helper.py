def test_policy(env, action, random=0):
    if random == 0:

        # Try the new policy to compute the number of times the agent
        # reaches the goal within a fixed number of steps in each episode

        # Set variable goal to 0
        goal = 0

        # Loop pver the number of episodes
        for i in range(100):

            # Reset the environment, this will return the initial state
            c = env.reset()

            # Loop over the maximum number of steps in each episode
            for t in range(10000):

                # Use the step method using the action sequence got from the toptimal policy
                c, reward, done, info = env.step(action[c])

                # If the reward returned by the environment is 1, then the goal is reached
                if done:
                    if reward == 1:
                        goal += 1
                    break

        print(" Agent succeeded to reach goal {} out of 100 episodes using the optimal policy ".format(goal))


    else:

        # Try the result with a random policy

        # Set variable goal to 0
        goal = 0

        # Loop pver the number of episodes
        for i in range(100):

            # Reset the environment, this will return the initial state
            c = env.reset()

            # Loop over the maximum number of steps in each episode
            for t in range(10000):

                # Use the step method using the action sequence got from the toptimal policy
                try:
                    c, reward, done, info = env.step(env.observation_space.sample())
                except:
                    c, reward, done, info = env.step(0)

                # If the reward returned by the environment is 1, then the goal is reached
                if done:
                    if reward == 1:
                        goal += 1
                    break

        print(" Agent succeeded to reach goal {} out of 100 episodes using a random policy ".format(goal))
    env.close()

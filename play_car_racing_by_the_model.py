import argparse
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
import matplotlib.pyplot as plt
import imageio 
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True,
                        help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1,
                        help='The number of episodes should the model plays.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes

    env = gym.make('CarRacing-v0')
    # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent = CarRacingDQNAgent(epsilon=0)
    agent.load(train_model)

    images = []#GIF
    training_results = []#matplot

    for e in range(play_episodes):
        init_state = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque(
            [init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1

        while True:
            env.render()
            
            image = env.render(mode='rgb_array')#GIF
            images.append(Image.fromarray(image))#GIF

            current_state_frame_stack = generate_state_frame_stack_from_queue(
                state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                training_results.append(total_reward)
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(
                    e+1, play_episodes, time_frame_counter, float(total_reward)))
                break
            time_frame_counter += 1
    imageio.mimsave('gameplay.gif', images, duration=0.1)#GIF origin 0.1
    plt.plot(training_results)
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Training Results')
    plt.show()
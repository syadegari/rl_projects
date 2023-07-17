import logging

import numpy as np
import torch

def log_scores(agent, i_episode, scores_window, score_threshold):
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window) >= score_threshold:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')


# import logging
# import numpy as np
# import torch

# # configure the logging module to display messages at or above the INFO level
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# def log_scores(agent, i_episode, scores_window, score_threshold):
#     avg_score = np.mean(scores_window)
#     logging.info('\rEpisode %s\tAverage Score: %.2f', i_episode, avg_score)
#     if i_episode % 100 == 0:
#         logging.info('\rEpisode %s\tAverage Score: %.2f', i_episode, avg_score)
#     if avg_score >= score_threshold:
#         logging.info('\nEnvironment solved in %s episodes!\tAverage Score: %.2f', i_episode-100, avg_score)
#         torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

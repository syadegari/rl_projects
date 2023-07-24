import logging
import numpy as np
import torch


class LogScores:
    def __init__(self):
        self.has_logged_solved_env = False

    def log_scores(self, agent, i_episode, scores_window, eps, score_threshold, experiment_name):
        log_line = f'\rEpisode {i_episode}\tAve. Score: {np.mean(scores_window):.2f} eps: {eps:.5f}'
        print(log_line, end="")
        if i_episode % 10 == 0:
            print(log_line)
            return False
        if np.mean(scores_window) >= score_threshold:
            if not self.has_logged_solved_env:
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                torch.save(agent.qnetwork_local.state_dict(), f'{experiment_name}_checkpoint.pth')
                self.has_logged_solved_env = True
            return True

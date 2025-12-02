from Trainer.PettingZoo.SB3.PPO import PPOTrainer
from Trainer.PettingZoo.SB3.A2C import A2CTrainer
from Trainer.PettingZoo.SB3.DQN import DQNTrainer
from Trainer.PettingZoo.SB3.RPPO import RPPOTrainer
from Environment.PettingZooEnv import (
    SETTING_FAIR_IN_ADVANTAGE, 
    SETTING_FULL_DETERMINISTIC, 
    SETTING_HALF_DETERMINISTIC, 
    SETTING_META_GAME_AWARE, 
    SETTING_RANDOM
)

N_MATCHES = 10_000

if __name__ == "__main__":
    trainers = [PPOTrainer(), A2CTrainer(), DQNTrainer(), RPPOTrainer()]
    for i, trainer in enumerate(trainers):
        for j, opponent in enumerate(trainers):
            if i == j: continue
            trainer.play_against(opponent, N_MATCHES, SETTING_META_GAME_AWARE)
            trainer.play_against(opponent, N_MATCHES, SETTING_FULL_DETERMINISTIC)
        
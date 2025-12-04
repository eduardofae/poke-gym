from stable_baselines3 import DQN
from Trainer.PettingZoo.SB3.SB3 import SB3Trainer
from Environment.PettingZooEnv import SelfPlayWrapper
import os

class DQNTrainer(SB3Trainer):
    def __init__(self, model_name='DQN', model_folder=None):
        super().__init__(model_name, model_folder)
        if os.path.isfile(self.model_path):
            self.model = DQN.load(self.model_path)
        else:
            self.model = self.get_default_model()

    def get_default_model(self):
        return DQN(
            policy="MlpPolicy",
            env=SelfPlayWrapper(),
            learning_rate=1e-4,
            buffer_size=200000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=1,
        )
    
if __name__ == "__main__":
    trainer = DQNTrainer()
    trainer.train()
    trainer.evaluate(50)
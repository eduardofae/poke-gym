from stable_baselines3 import PPO
from Trainer.PettingZoo.SB3.SB3 import SB3Trainer
from Environment.PettingZooEnv import SelfPlayWrapper
import os

class PPOTrainer(SB3Trainer):
    def __init__(self, model_name='PPO', model_folder=None):
        super().__init__(model_name, model_folder)
        if os.path.isfile(self.model_path):
            self.model = PPO.load(self.model_path)
        else:
            self.model = self.get_default_model()
        
    def get_default_model(self):
        return PPO(
            "MlpPolicy",
            learning_rate=3e-4,
            env=SelfPlayWrapper(),
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
        )
    
if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
    trainer.evaluate(50)
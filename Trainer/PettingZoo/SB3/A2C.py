from stable_baselines3 import A2C
from Trainer.PettingZoo.SB3.SB3 import SB3Trainer
from Environment.PettingZooEnv import SelfPlayWrapper
import os

class A2CTrainer(SB3Trainer):
    def __init__(self):
        super().__init__('A2C')
        if os.path.isfile(self.model_path+'.zip'):
            self.model = A2C.load(self.model_path)
        else:
            self.model = self.get_default_model()
        

    def get_default_model(self):
        return A2C(
            policy="MlpPolicy",
            env=SelfPlayWrapper(),
            learning_rate=7e-4,
            gamma=0.99,
            n_steps=5,
            ent_coef=0.01,
            verbose=1,
        )
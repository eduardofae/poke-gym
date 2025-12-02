from sb3_contrib import RecurrentPPO
from Trainer.PettingZoo.SB3.SB3 import SB3Trainer
from Environment.PettingZooEnv import SelfPlayWrapper
import os

class RPPOTrainer(SB3Trainer):
    def __init__(self):
        super().__init__('RPPO')
        if os.path.isfile(self.model_path+'.zip'):
            self.model = RecurrentPPO.load(self.model_path)
        else:
            self.model = self.get_default_model()
        

    def get_default_model(self):
        return RecurrentPPO(
            policy="MlpLstmPolicy",
            env=SelfPlayWrapper(),
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            verbose=1
        )
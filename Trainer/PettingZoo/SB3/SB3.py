from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

from Environment.PettingZooEnv import (
    SelfPlayWrapper,
    SETTING_META_GAME_AWARE,
    SETTING_FULL_DETERMINISTIC
)

MODEL_FOLDER = '.\\Model\\PettingZoo\\SB3'

class SB3Trainer():
    def __init__(self, model_name, model_folder=None):
        self.model_name = model_name
        if model_folder is None:
            model_folder = MODEL_FOLDER
        self.model_path = f'{model_folder}\\{model_name}_pkm_model.zip'
        self.checkpoint_folder = f'{model_folder}\\{model_name}_checkpoints\\'

    def train(self, model=None):
        def make_env():
            return Monitor(SelfPlayWrapper(opponent_policy=model, setting=SETTING_META_GAME_AWARE))
        
        env = DummyVecEnv([make_env])

        callback = CheckpointCallback(save_freq=100000, save_path=self.checkpoint_folder)

        self.model.learn(total_timesteps=1_000_000, callback=callback)
        self.model.save(self.model_path)

        env.close()

    def evaluate(self, n_matches=1, setting=SETTING_FULL_DETERMINISTIC, model=None):
        """
        Evaluation in the fully deterministic Pokémon configuration.
        Opponent = frozen copy of trained model.
        """

        def make_eval_env():
            return Monitor(
                SelfPlayWrapper(
                    opponent_policy=model,
                    setting=setting,
                )
            )

        env = DummyVecEnv([make_eval_env])

        mean_reward, std_reward = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_matches,
            deterministic=True
        )

        print("\n=====================================")
        print(f" {self.model_name} Evaluation ({n_matches})")
        print("=====================================")
        print(f" Mean reward: {mean_reward}")
        print(f" Std reward : {std_reward}")
        print("=====================================\n")

        env.close()
    
    def get_action(self, obs):
        action, _ = self.model.predict(
            obs,
            deterministic=True
        )
        return action
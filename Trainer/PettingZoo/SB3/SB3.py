from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

from Environment.PettingZooEnv import (
    SelfPlayWrapper,
    SimplePkmEnv,
    SETTING_META_GAME_AWARE,
    SETTING_FULL_DETERMINISTIC,
    SETTING_TO_STR
)

MODEL_FOLDER = '.\\Model\\PettingZoo\\SB3'

class SB3Trainer():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = f'{MODEL_FOLDER}\\{model_name}_pkm_model'
        self.checkpoint_folder = f'{MODEL_FOLDER}\\{model_name}_checkpoints\\'

    def train(self):
        def make_env():
            return Monitor(SelfPlayWrapper(setting=SETTING_META_GAME_AWARE))
        
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
    
    def play_against(self, trainer, n_matches, setting=SETTING_META_GAME_AWARE):
        env = SimplePkmEnv(setting)
        victories = 0
        draws = 0
        model = trainer.model
        for _ in range(n_matches):
            obs, _ = env.reset()
            while True:
                player = env.agents[0]
                opponent = env.agents[1]
                action, _ = self.model.predict(
                    obs[player],
                    deterministic=True
                )
                opp_action, _ = model.predict(
                    obs[opponent],
                    deterministic=True
                )
                actions = {
                    player: int(action),
                    opponent: int(opp_action)
                }
                obs_dict, rewards, terms, truncs, _ = env.step(actions)
                
                if terms[player]:
                    victories += rewards[player]
                    break

                if truncs[player]:
                    draws += 1
                    break
        
        defeats = n_matches - (draws + victories)
        print("\n=====================================")
        print(f" {self.model_name} X {trainer.model_name} ({SETTING_TO_STR[setting]}) ({n_matches})")
        print("=====================================")
        print(f" Victories: {victories} ({victories/n_matches*100:.2f}%)")
        print(f" Draws: {draws} ({draws/n_matches*100:.2f}%)")
        print(f" Defeats: {defeats} ({defeats/n_matches*100:.2f}%)")
        print("=====================================\n")


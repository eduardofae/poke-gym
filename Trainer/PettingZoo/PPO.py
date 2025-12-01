from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from Environment.PettingZooEnv import (
    SelfPlayWrapper,
    SETTING_META_GAME_AWARE,
    SETTING_FULL_DETERMINISTIC,
)

MODEL_PATH = '.\\Model\\Deep\\SB3-PPO'

def train():
    def make_env():
        return Monitor(SelfPlayWrapper(setting=SETTING_META_GAME_AWARE))

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=300_000)
    model.save(MODEL_PATH + "ppo_pkm_selfplay.zip")
    print("\nModel saved to ppo_pkm_selfplay.zip")

    return model

def evaluate(model):
    """
    Evaluation in the fully deterministic Pokémon configuration.
    Opponent = frozen copy of trained model.
    """

    def make_eval_env():
        return Monitor(
            SelfPlayWrapper(
                opponent_policy=None,
                setting=SETTING_FULL_DETERMINISTIC,
            )
        )

    eval_env = DummyVecEnv([make_eval_env])

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=50,
        deterministic=True
    )

    print("\n=====================================")
    print(" PPO Evaluation (Deterministic Env)")
    print("=====================================")
    print(f" Mean reward: {mean_reward}")
    print(f" Std reward : {std_reward}")
    print("=====================================\n")


if __name__ == "__main__":
    print("Training PPO in META_GAME_AWARE setting...")
    model = train()

    print("\nEvaluating PPO in FULL_DETERMINISTIC setting...")
    evaluate(model)

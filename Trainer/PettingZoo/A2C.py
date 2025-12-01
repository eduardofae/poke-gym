from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

from Environment.PettingZooEnv import (
    SelfPlayWrapper,
    SETTING_META_GAME_AWARE,
    SETTING_FULL_DETERMINISTIC,
)

MODEL_PATH = '.\\Model\\PettingZoo\\SB3\\'

def train():
    def make_env():
        return Monitor(SelfPlayWrapper(setting=SETTING_META_GAME_AWARE))
    
    env = make_env()

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,
        gamma=0.99,
        n_steps=5,
        ent_coef=0.01,
        verbose=1,
    )

    callback = CheckpointCallback(save_freq=100000, save_path=MODEL_PATH + "a2c_checkpoints\\")

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save(MODEL_PATH + "a2c_pkm_model")

    env.close()

    return model

def evaluate(model=MODEL_PATH + "a2c_pkm_model"):
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

    env = DummyVecEnv([make_eval_env])

    if isinstance(model, str):
        model = A2C.load(model, env=env)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=200,
        deterministic=True
    )

    print("\n=====================================")
    print(" A2C Evaluation (Deterministic Env)")
    print("=====================================")
    print(f" Mean reward: {mean_reward}")
    print(f" Std reward : {std_reward}")
    print("=====================================\n")

    env.close()


if __name__ == "__main__":
    print("Training A2C in META_GAME_AWARE setting...")
    model = train()

    print("\nEvaluating A2C in FULL_DETERMINISTIC setting...")
    evaluate(model)

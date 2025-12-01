from sb3_contrib import RecurrentPPO
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

    env = DummyVecEnv([make_env])

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        verbose=1
    )

    callback = CheckpointCallback(save_freq=100000, save_path=MODEL_PATH + "rppo_checkpoints\\")

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save(MODEL_PATH + "rppo_pkm_model")
    print("\nModel saved to ppo_pkm_selfplay.zip")

    env.close()

    return model


def evaluate(model=MODEL_PATH + "dqn_pkm_model"):
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
        model = RecurrentPPO.load(model, env=env)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=200,
        deterministic=True
    )

    print("\n=====================================")
    print(" RPPO Evaluation (Deterministic Env)")
    print("=====================================")
    print(f" Mean reward: {mean_reward}")
    print(f" Std reward : {std_reward}")
    print("=====================================\n")

    env.close()


if __name__ == "__main__":
    print("Training RPPO in META_GAME_AWARE setting...")
    model = train()

    print("\nEvaluating RPPO in FULL_DETERMINISTIC setting...")
    evaluate(model)

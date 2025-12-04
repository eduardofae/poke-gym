from Trainer.PettingZoo.SB3.PPO import PPOTrainer
from Trainer.PettingZoo.SB3.A2C import A2CTrainer
from Trainer.PettingZoo.SB3.DQN import DQNTrainer
from Trainer.PettingZoo.SB3.RPPO import RPPOTrainer
from Environment.PettingZooEnv import (
    SimplePkmEnv,
    SETTING_TO_STR,
    SETTING_FAIR_IN_ADVANTAGE, 
    SETTING_FULL_DETERMINISTIC, 
    SETTING_HALF_DETERMINISTIC, 
    SETTING_META_GAME_AWARE, 
    SETTING_RANDOM
)

N_GAMES = 10_000

def get_action(model, player, env, obs):
    if model is None:
        return env.action_space(player).sample()
    return model.get_action(obs[player])

def play_match(trainer1=None, trainer2=None, n_games=N_GAMES, setting=SETTING_META_GAME_AWARE):
        env = SimplePkmEnv(setting)
        victories = 0
        draws = 0
        for _ in range(n_games):
            obs, _ = env.reset()
            while True:
                player, opponent = env.agents
                player_action = get_action(trainer1, player, env, obs)
                opponent_action = get_action(trainer2, opponent, env, obs)
                actions = {
                    player: int(player_action),
                    opponent: int(opponent_action)
                }
                obs_dict, rewards, terms, truncs, _ = env.step(actions)
                if terms[player]:
                    if rewards[player] >= 0.:
                        victories += 1
                    break
                if truncs[player]:
                    draws += 1
                    break
        defeats = n_games - (draws + victories)
        t1_name = trainer1.model_name if trainer1 is not None else 'RANDOM'
        t2_name = trainer2.model_name if trainer2 is not None else 'RANDOM'
        print("\n=====================================")
        print(f" {t1_name} X {t2_name} ({SETTING_TO_STR[setting]}) ({n_games})")
        print("=====================================")
        print(f" Victories: {victories} ({victories/n_games*100:.2f}%)")
        print(f" Draws: {draws} ({draws/n_games*100:.2f}%)")
        print(f" Defeats: {defeats} ({defeats/n_games*100:.2f}%)")
        print("=====================================\n")
        env.close()

OLD_MODEL_FOLDER = '.\\Model\\PettingZoo\\old_reward\\SB3'
if __name__ == "__main__":
    old_trainers = [PPOTrainer('PPO-old', OLD_MODEL_FOLDER), A2CTrainer('A2C-old', OLD_MODEL_FOLDER), DQNTrainer('DQN-old', OLD_MODEL_FOLDER), RPPOTrainer('RPPO-old', OLD_MODEL_FOLDER)]
    new_trainers = [PPOTrainer('PPO-new'), A2CTrainer('A2C-new'), DQNTrainer('DQN-new'), RPPOTrainer('RPPO-new')]
    for i, trainer in enumerate(new_trainers):
        opponent = old_trainers[i]
        play_match(trainer, opponent, N_GAMES, SETTING_META_GAME_AWARE)
        play_match(opponent, trainer, N_GAMES, SETTING_META_GAME_AWARE)
        play_match(trainer, opponent, N_GAMES, SETTING_FULL_DETERMINISTIC)
        play_match(opponent, trainer, N_GAMES, SETTING_FULL_DETERMINISTIC)
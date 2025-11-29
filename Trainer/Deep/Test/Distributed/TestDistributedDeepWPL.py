from Environment.SimplePkmEnv import *
from Trainer.Deep.Learning.Distributed.DistributedDeepWPL import *
import sys

G_L_RATE = 1e-4
PI_L_RATE = 1 / 200  # 1 / 200
Y = 0.9
TAU = 25  # BATCH_SIZE
N_EPS = 200000
N_STEPS = TAU
E_RATE = 1.0
MIN_E_RATE = [0.5, 0.1, 0.01]
N_PLAYERS = 2
DECAY_PERCENTAGE = 0.65
ENV_NAME = 'SimplePkmEnv(SETTING_HALF_DETERMINISTIC)'
# MODEL_PATH = '../../../../Model/Deep/DistributedDeepWPL' + '_' + ENV_NAME
# MODEL_PATH = r'C:\Users\barbu\VSCode-projects\poke-gym\Model\Deep\DistributedDeepWPL'
MODEL_PATH = '.\\Model\\Deep\\DistributedDeepWPL'

def main():
    task_index = int(sys.argv[1])
    concurrent_games = int(sys.argv[2])
    url = "localhost"
    hosts = [url + ":" + str(2210 + i) for i in range(concurrent_games)]
    env = SimplePkmEnv(SETTING_HALF_DETERMINISTIC)
    trainer = DistributedDeepWPL()
    print('train', task_index)
    trainer.train(env=env,
                  g_l_rate=G_L_RATE,
                  concurrent_games=concurrent_games,
                  pi_l_rate=PI_L_RATE, y=Y,
                  tau=TAU,
                  n_eps=N_EPS,
                  n_steps=N_STEPS,
                  e_rate=E_RATE,
                  n_players=N_PLAYERS,
                  model_path=MODEL_PATH,
                  decay_percentage=DECAY_PERCENTAGE,
                  min_e_rate=MIN_E_RATE,
                  hosts=hosts,
                  task_index=task_index)

if __name__ == '__main__':
    main()

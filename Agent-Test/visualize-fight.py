# play_vs_agent.py
import os, sys, pathlib, argparse
import numpy as np
import tensorflow as tf
from collections import deque

# --- TF1 graph mode must come first ---
tf.compat.v1.disable_eager_execution()

# repo path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from Environment.SimplePkmEnv import *
from Trainer.Deep.Learning.Distributed.DistributedDeepWPL import DistributedDeepWPL
from Trainer.Deep.Learning.Distributed.DistributedDeepGIGAWoLF import DistributedDeepGIGAWoLF

MODEL_PATH = r"C:\Users\barbu\VSCode-projects\poke-gym\Model\Deep\DistributedDeepGigaWolf"
# MODEL_PATH = r"C:\Users\barbu\VSCode-projects\poke-gym\Model\Deep\DistributedDeepGigaWolf"

TOPK = 5            # how many top policy actions to print
SHOW_Q = False       # print Q-values
SHOW_PI = True      # print top-k policy probs
SMOOTH_REWARD = 50  # moving average window

# ---------- Pretty printing helpers ----------
def bar(val, width=20, filled='█', empty='·'):
    """Render a simple bar for values in [0,1]."""
    val = float(np.clip(val, 0.0, 1.0))
    n = int(round(val * width))
    return filled * n + empty * (width - n)

def action_label(a: int, env) -> str:
    """
    Map numeric action -> short human label.
    Customize this to your env action semantics.
    """
    # If your env exposes human-readable names, plug them here:
    # if hasattr(env, "action_meanings"):
    #     try: return env.action_meanings()[a]
    #     except: pass
    return f"A{a}"

def summarize_obs(obs):
    """
    Turn the raw observation into something readable.
    If your obs is already a flat vector in [0,1], interpret first dims as HP/energy, etc.
    Adjust to your env. Keeps a safe fallback.
    """
    obs = np.asarray(obs, dtype=np.float32).ravel()
    txt = []
    if obs.size >= 2:
        hp = np.clip(obs[0], 0, 1)
        enr = np.clip(obs[1], 0, 1)
        txt.append(f"HP  [{bar(hp)}] {hp:0.2f}")
        txt.append(f"ENR [{bar(enr)}] {enr:0.2f}")
    else:
        txt.append(str(obs))
    return " | ".join(txt)

def print_turn_header(step):
    print("\n" + "=" * 64)
    print(f"TURN {step}")
    print("=" * 64)

# def print_player_state(player_idx, obs):
    # print(f"Player {player_idx} obs: {summarize_obs(obs)}")
    # pass

def print_player_state(player_idx, obs, env: SimplePkmEnv):
    p = env.a_pkm[player_idx]
    hp_ratio = p.hp / HIT_POINTS
    types_str = TYPE_TO_STR[p.p_type[0]]
    if p.p_type[1] != NONE:
        types_str += "/" + TYPE_TO_STR[p.p_type[1]]
    print(f"Player {player_idx} [{types_str}] HP [{bar(hp_ratio)}] {p.hp:0.1f}/{HIT_POINTS}")


def print_agent_diagnostics(pi, q, env):
    if SHOW_PI:
        top_idx = np.argsort(-pi)[:TOPK]
        top_str = ", ".join([f"{action_label(i, env)}:{pi[i]:.2f}" for i in top_idx])
        print(f"Agent π top-{TOPK}: {top_str}")
    if SHOW_Q:
        # show min, max and the chosen top too
        print(f"Agent Q: min={np.min(q):.3f}  max={np.max(q):.3f}")

def choose_human_action(env, player_idx, obs):
    print(f"\nYour turn (player {player_idx}).")
    n_actions = int(env.action_space.n)
    while True:
        raw = input(f"Choose action [0..{n_actions-1}] ({', '.join(action_label(i, env) for i in range(n_actions))}): ").strip()
        if raw.isdigit():
            a = int(raw)
            if 0 <= a < n_actions:
                return a
        print("Invalid input.")

# ---------- TF bits ----------
def build_agent_network_wpl(env):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    # IMPORTANT: name/scope must match training
    return DistributedDeepWPL.Network(optimizer, env, name='global_0', global_id=0)

def build_agent_network_giga_wolf(env):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    # IMPORTANT: name/scope must match training
    return DistributedDeepGIGAWoLF.Network(optimizer, env, name='global_0', global_id=0)

def try_load_weights_v1_checkpoint(sess, model_path):
    ckpt = tf.train.latest_checkpoint(model_path)
    if not ckpt:
        print("[load v1] no checkpoint found in:", model_path)
        return False
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
    saver.restore(sess, ckpt)
    print("[load v1] restored from:", ckpt)
    return True

def agent_action_from_net_v1(net, obs, sess):
    q, pi = sess.run([net.out_q, net.policy], feed_dict={net.in_state: [obs]})
    q  = q[0]
    pi = pi[0]
    a  = np.random.choice(len(pi), p=pi)
    return a, q, pi

def parse_args():
    parser = argparse.ArgumentParser(description="Play vs agent with selectable SimplePkmEnv setting")
    parser.add_argument(
        "--setting",
        type=str,
        default="half",
        help="Environment setting: name (random|full|half|fair|meta) or code (0|1|2|3|4). Default: half",
    )
    return parser.parse_args()

def resolve_setting(setting_str):
    name_map = {
        "random": SETTING_RANDOM,
        "full": SETTING_FULL_DETERMINISTIC,
        "half": SETTING_HALF_DETERMINISTIC,
        "fair": SETTING_FAIR_IN_ADVANTAGE,
        "meta": SETTING_META_GAME_AWARE
    }
    # allow numeric codes
    try:
        code = int(setting_str)
        if code in (SETTING_RANDOM, SETTING_FULL_DETERMINISTIC, SETTING_HALF_DETERMINISTIC, SETTING_FAIR_IN_ADVANTAGE, SETTING_META_GAME_AWARE):
            return code
    except ValueError:
        pass
    # fallback to names
    setting_str = setting_str.lower()
    if setting_str in name_map:
        return name_map[setting_str]
    raise ValueError("Invalid --setting. Use name (random|full|half|fair) or code (0|1|2|3).")

# ---------- Main loop ----------
def main():
    args = parse_args()
    setting_id = resolve_setting(args.setting)

    env = SimplePkmEnv(setting=setting_id)
    print(f"[info] Using env setting={setting_id} (passed --setting={args.setting})")

    # net = build_agent_network_wpl(env)
    net = build_agent_network_giga_wolf(env)

    rewards_ma = deque(maxlen=SMOOTH_REWARD)
    cum_reward = np.array([0.0, 0.0], dtype=np.float32)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        loaded = try_load_weights_v1_checkpoint(sess, MODEL_PATH)
        print("[info] Weights loaded:", loaded)

        obs = env.reset()
        done = False
        step = 0
        print("\nStarting match: You (P0) vs Agent (P1). Ctrl+C to quit.")
        while not done:
            step += 1
            obs_p0, obs_p1 = obs

            print_turn_header(step)
            print_player_state(0, obs_p0, env)
            print_player_state(1, obs_p1, env)

            if hasattr(env, "render"):
                try:
                    env.render()
                except Exception:
                    pass

            a0 = choose_human_action(env, 0, obs_p0)
            a1, q1, pi1 = agent_action_from_net_v1(net, obs_p1, sess)

            print(f"\nChosen actions -> P0: {action_label(a0, env)}  |  P1: {action_label(a1, env)}")
            print_agent_diagnostics(pi1, q1, env)

            obs, reward, done, info = env.step((a0, a1))
            rewards_ma.append(reward)
            cum_reward += np.asarray(reward, dtype=np.float32)

            r_str = f"P0:{reward[0]:+.3f}  P1:{reward[1]:+.3f}"
            cr_str = f"P0:{cum_reward[0]:+.3f}  P1:{cum_reward[1]:+.3f}"
            if len(rewards_ma) > 0:
                ma = np.mean(np.vstack(rewards_ma), axis=0)
                ma_str = f"P0:{ma[0]:+.3f}  P1:{ma[1]:+.3f}"
            else:
                ma_str = "n/a"
            print(f"Reward this turn: {r_str}")
            print(f"Cumulative      : {cr_str}")
            print(f"Moving avg({SMOOTH_REWARD}) : {ma_str}")

        print("\nEpisode finished.")
        print(f"Final cumulative reward  -> P0:{cum_reward[0]:+.3f}  P1:{cum_reward[1]:+.3f}")

if __name__ == "__main__":
    main()

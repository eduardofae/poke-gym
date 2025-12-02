import functools

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box
from copy import copy
import time

from pettingzoo import ParallelEnv

# Avoid infinite games due to constant switching
MAX_CONSECUTIVE_SWITCHES = 4
MAX_CONSECUTIVE_ZERO_DMG_DEALT = 4
SMALL_DAMAGE_THRESHOLD = 1e-4

# type codification
NONE = -1
NORMAL = 0
FIRE = 1
WATER = 2
ELECTRIC = 3
GRASS = 4
ICE = 5
FIGHT = 6
POISON = 7
GROUND = 8
FLYING = 9
PSYCHIC = 10
BUG = 11
ROCK = 12
GHOST = 13
DRAGON = 14
DARK = 15
STEEL = 16
FAIRY = 17

TYPE_TO_STR = {NONE: "NONE", NORMAL: "NORMAL", FIRE: "FIRE", WATER: "WATER", ELECTRIC: "ELECTRIC", GRASS: "GRASS", ICE: "ICE",
               FIGHT: "FIGHT", POISON: "POISON", GROUND: "GROUND", FLYING: "FLYING", PSYCHIC: "PSYCHIC", BUG: "BUG",
               ROCK: "ROCK", GHOST: "GHOST", DRAGON: "DRAGON", DARK: "DARK", STEEL: "STEEL", FAIRY: "FAIRY"}
TYPE_LIST = np.array([NONE, NORMAL, FIRE, WATER, ELECTRIC, GRASS, ICE, FIGHT, POISON, GROUND, FLYING, PSYCHIC, BUG, ROCK, GHOST,
             DRAGON, DARK, STEEL, FAIRY])
N_TYPES = len(TYPE_LIST)

# type chart
TYPE_CHART_MULTIPLIER = [
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., .5, .0, 1., 1., .5, 1.],  # NORMAL
    [1., .5, .5, 1., 2., 2., 1., 1., 1., 1., 1., 2., .5, 1., .5, 1., 2., 1.],  # FIRE
    [1., 2., .5, 1., .5, 1., 1., 1., 2., 1., 1., 1., 2., 1., .5, 1., 1., 1.],  # WATER
    [1., 1., 2., .5, .5, 1., 1., 1., 0., 2., 1., 1., 1., 1., .5, 1., 1., 1.],  # ELECTRIC
    [1., .5, 2., 1., .5, 1., 1., .5, 2., .5, 1., .5, 2., 1., .5, 1., .5, 1.],  # GRASS
    [1., .5, .5, 1., 2., .5, 1., 1., 2., 2., 1., 1., 1., 1., 2., 1., .5, 1.],  # ICE
    [2., 1., 1., 1., 1., 2., 1., .5, 1., .5, .5, .5, 2., 0., 1., 2., 2., .5],  # FIGHTING
    [1., 1., 1., 1., 2., 1., 1., .5, .5, 1., 1., 1., .5, .5, 1., 1., .0, 2.],  # POISON
    [1., 2., 1., 2., .5, 1., 1., 2., 1., 0., 1., .5, 2., 1., 1., 1., 2., 1.],  # GROUND
    [1., 1., 1., .5, 2., 1., 2., 1., 1., 1., 1., 2., .5, 1., 1., 1., .5, 1.],  # FLYING
    [1., 1., 1., 1., 1., 1., 2., 2., 1., 1., .5, 1., 1., 1., 1., 0., .5, 1.],  # PSYCHIC
    [1., .5, 1., 1., 2., 1., .5, .5, 1., .5, 2., 1., 1., .5, 1., 2., .5, .5],  # BUG
    [1., 2., 1., 1., 1., 2., .5, 1., .5, 2., 1., 2., 1., 1., 1., 1., .5, 1.],  # ROCK
    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 2., 1., .5, 1., 1.],  # GHOST
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., .5, .0],  # DRAGON
    [1., 1., 1., 1., 1., 1., .5, 1., 1., 1., 2., 1., 1., 2., 1., .5, 1., .5],  # DARK
    [1., .5, .5, .5, 1., 2., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., .5, 2.],  # STEEL
    [1., .5, 1., 1., 1., 1., 2., .5, 1., 1., 1., 1., 1., 1., 2., 2., .5, 1.]   # FAIRY
]

# move power range
POWER_MIN = 50
POWER_MAX = 100

# number moves
N_MOVES = 4
SWITCH_ACTION = N_MOVES

# hit Points
HIT_POINTS = POWER_MAX + POWER_MIN + 150.

# settings
SETTING_RANDOM = 0
SETTING_FULL_DETERMINISTIC = 1
SETTING_HALF_DETERMINISTIC = 2
SETTING_FAIR_IN_ADVANTAGE = 3
SETTING_META_GAME_AWARE = 4

SETTING_TO_STR = {
    SETTING_RANDOM: "RANDOM", 
    SETTING_FULL_DETERMINISTIC: "FULL_DETERMINISTIC", 
    SETTING_HALF_DETERMINISTIC: "HALF_DETERMINISTIC", 
    SETTING_FAIR_IN_ADVANTAGE: "FAIR_IN_ADVANTAGE",
    SETTING_META_GAME_AWARE: "META_GAME_AWARE"
}

class SelfPlayWrapper(gym.Env):
    """
    Converts a 2-player Parallel PettingZoo environment into a single-agent
    Gymnasium environment compatible with Stable-Baselines3.

    - Agent controls player_0
    - Opponent (player_1) uses either a random policy or a fixed SB3 policy
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_policy=None, setting=SETTING_META_GAME_AWARE):
        super().__init__()

        self.env = SimplePkmEnv(setting=setting)
        self.opponent_policy = opponent_policy  # SB3 model or None (random)

        # Get sample obs to determine space dimensions
        sample_obs, _ = self.env.reset()
        obs0 = sample_obs["player_0"]

        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=obs0.shape, dtype=np.float32
        )
        self.action_space = self.env.action_space("player_0")

        self.last_opp_obs = None

    def reset(self, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed)
        self.last_opp_obs = obs_dict["player_1"]
        return obs_dict["player_0"].astype(np.float32), info

    def step(self, action):
        # Opponent action
        if self.opponent_policy is None:
            opp_action = np.random.randint(self.action_space.n)
        else:
            opp_action, _ = self.opponent_policy.predict(
                np.array(self.last_opp_obs, dtype=np.float32),
                deterministic=True
            )

        actions = {
            "player_0": int(action),
            "player_1": int(opp_action)
        }

        obs_dict, rewards, terms, truncs, infos = self.env.step(actions)

        # Save next opponent obs
        if "player_1" in obs_dict:
            self.last_opp_obs = obs_dict["player_1"]

        return (
            obs_dict["player_0"].astype(np.float32),
            rewards["player_0"],
            terms["player_0"],
            truncs["player_0"],
            infos,
        )

class SimplePkmEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "SimplePkmEnv"}

    def __init__(self, setting=SETTING_RANDOM, debug=False, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(2)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.setting = setting
        self.debug = debug
        self.debug_message = ['', '']
        self.a_pkm = [SimplePkm(), SimplePkm()]  # active pokemons
        self.p_pkm = [SimplePkm(), SimplePkm()]  # party pokemons

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # Discrete(4) means an integer in range(0, 4)
        return Discrete(len(encode(self._state_trainer(0))))

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(N_MOVES + 1)

    def render(self, view='general'):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.debug:
            if self.debug_message[0] != '' and self.debug_message[1] != '':
                if self.switched[0]:
                    if self.has_fainted:
                        print('Trainer 1', self.debug_message[1])
                        print('Trainer 0', self.debug_message[0])
                    else:
                        print('Trainer 0', self.debug_message[0])
                        print('Trainer 1', self.debug_message[1])
                elif self.switched[1]:
                    if self.has_fainted:
                        print('Trainer 0', self.debug_message[0])
                        print('Trainer 1', self.debug_message[1])
                    else:
                        print('Trainer 1', self.debug_message[1])
                        print('Trainer 0', self.debug_message[0])
                elif self.first == 0:
                    print('Trainer 0', self.debug_message[0])
                    print('Trainer 1', self.debug_message[1])
                else:
                    print('Trainer 1', self.debug_message[1])
                    print('Trainer 0', self.debug_message[0])
            print()
        print('Trainer 0')
        print('Active', self.a_pkm[0])
        print('Party', self.p_pkm[0])
        print('Trainer 1')
        print('Active', self.a_pkm[1])
        if view != 'player':
            print('Party', self.p_pkm[1])
        print()

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = copy(self.possible_agents)
        self.num_moves = 0
        self.consecutive_switches = 0
        self.consecutive_no_damage = 0

        if self.setting == SETTING_RANDOM:
            self.a_pkm = [SimplePkm(), SimplePkm()]  # active pokemons
            self.p_pkm = [SimplePkm(), SimplePkm()]  # party pokemons
        elif self.setting == SETTING_FULL_DETERMINISTIC:
            self.a_pkm = [SimplePkm(GRASS, HIT_POINTS, GRASS, 90, FIRE, 90, GRASS, 90, FIRE, 90),
                          SimplePkm(FIRE, HIT_POINTS, FIRE, 90, FIRE, 90, FIRE, 90, FIRE, 90)]  # active pokemons
            self.p_pkm = [SimplePkm(WATER, HIT_POINTS, FIGHT, 90, NORMAL, 90, NORMAL, 90, WATER, 90),
                          SimplePkm(NORMAL, HIT_POINTS, NORMAL, 90, NORMAL, 90, NORMAL, 90, NORMAL,
                                    90)]  # party pokemons
        elif self.setting == SETTING_HALF_DETERMINISTIC:
            if np.random.uniform(0, 1) <= 0.2:
                type1 = get_random_type_combo()
                type2 = get_random_type_combo(type1)
                self.a_pkm = [SimplePkm(type1, HIT_POINTS, type1[0], 90, type2[0], 90, type1[1], 90, type2[1], 90),
                              SimplePkm(type2, HIT_POINTS, type2[0], 90, type2[0], 90, type2[1], 90, type2[1],
                                        90)]  # active pokemons
            else:
                self.a_pkm = [SimplePkm(), SimplePkm()]  # active pokemons
            self.p_pkm = [SimplePkm(), SimplePkm()]  # party pokemons
        elif self.setting == SETTING_FAIR_IN_ADVANTAGE:
            type1 = get_random_type_combo()
            type2 = get_random_type_combo(type1)
            self.a_pkm = [
                SimplePkm(type1, get_non_very_effective_move(type2), 90, get_non_very_effective_move(type2), 90,
                          get_normal_effective_move(type2), 90, type1, 90),
                SimplePkm(type2, get_super_effective_move(type1), 90, get_non_very_effective_move(type1), 90,
                          get_normal_effective_move(type1), 90, type2, 90)]  # active pokemons
            self.p_pkm = [SimplePkm(), SimplePkm()]  # party pokemons
        elif self.setting == SETTING_META_GAME_AWARE:
            pokeA1_type = get_random_type_combo()
            pokeA1_moves = get_meta_aware_moves(pokeA1_type)
            pokeA2_type = get_random_type_combo()
            pokeA2_moves = get_meta_aware_moves(pokeA2_type)
            pokeP1_type = get_random_type_combo()
            pokeP1_moves = get_meta_aware_moves(pokeP1_type)
            pokeP2_type = get_random_type_combo()
            pokeP2_moves = get_meta_aware_moves(pokeP2_type)
            self.a_pkm = [
                SimplePkm(pokeA1_type, HIT_POINTS, pokeA1_moves[0], 90, pokeA1_moves[1], 90,
                          pokeA1_moves[2], 90, pokeA1_moves[3], 90),
                SimplePkm(pokeA2_type, HIT_POINTS, pokeA2_moves[0], 90, pokeA2_moves[1], 90,
                          pokeA2_moves[2], 90, pokeA2_moves[3], 90)]
            self.p_pkm = [
                SimplePkm(pokeP1_type, HIT_POINTS, pokeP1_moves[0], 90, pokeP1_moves[1], 90,
                          pokeP1_moves[2], 90, pokeP1_moves[3], 90),
                SimplePkm(pokeP2_type, HIT_POINTS, pokeP2_moves[0], 90, pokeP2_moves[1], 90,
                          pokeP2_moves[2], 90, pokeP2_moves[3], 90)]
    
        observations = {
            agent: np.array(encode(self._state_trainer(i))) 
            for i, agent in enumerate(self.agents)
        }
        infos = { agent: {} for agent in self.agents }
        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        self.has_fainted = False
        self.switched = [False, False]
        r = {agent: 0. for agent in self.agents}
        # switch pokemon
        if actions[self.agents[0]] == SWITCH_ACTION:
            if not SimplePkmEnv._fainted_pkm(self.p_pkm[0]):
                self._switch_pkm(0)
        if actions[self.agents[1]] == SWITCH_ACTION:
            if not SimplePkmEnv._fainted_pkm(self.p_pkm[1]):
                self._switch_pkm(1)

        # pokemon attacks
        order = [0, 1]
        # random attack order
        np.random.shuffle(order)
        self.first = order[0]
        self.second = order[1]
        action1 = actions[self.agents[self.first]]
        action2 = actions[self.agents[self.second]]

        terminal = False
        can_player2_attack = True

        # first attack
        dmg_dealt1 = 0
        dmg_dealt2 = 0

        if action1 != SWITCH_ACTION:
            r[self.agents[self.first]], terminal, can_player2_attack, dmg_dealt1 = self._battle_pkm(action1, self.first)

        if can_player2_attack and action2 != SWITCH_ACTION:
            r[self.agents[self.second]], terminal, _, dmg_dealt2 = self._battle_pkm(action2, self.second)
        elif self.debug:
            self.debug_message[self.second] = 'can\'t perform any action'

        r[self.agents[self.first]] -= dmg_dealt2 / HIT_POINTS
        r[self.agents[self.second]] -= dmg_dealt1 / HIT_POINTS        

        if dmg_dealt1 + dmg_dealt2 - SMALL_DAMAGE_THRESHOLD <= 0:
            self.consecutive_no_damage += 1
        else:
            self.consecutive_no_damage = 0
        if self.switched[0] and self.switched[1]:
            self.consecutive_switches +=1
        else:
            self.consecutive_switches = 0

        terminations = { agent: terminal for agent in self.agents }
        truncations = { agent: self.consecutive_switches >= MAX_CONSECUTIVE_SWITCHES or self.consecutive_no_damage >= MAX_CONSECUTIVE_ZERO_DMG_DEALT for agent in self.agents }

        observations = {
            agent: np.array(encode(self._state_trainer(i)))
            for i, agent in enumerate(self.agents)
        }

        infos = {agent: {} for agent in self.agents}

        if terminal:
            first_won = SimplePkmEnv._fainted_pkm(self.a_pkm[self.first])
            r[self.agents[self.first]] = 1 if first_won else 0
            r[self.agents[self.second]] = 0 if first_won else 1
            self.agents = []

        return observations, r, terminations, truncations, infos
    
    def change_setting(self, setting):
        self.setting = setting

    def _state_trainer(self, t_id):
        return [self.a_pkm[t_id].p_type,
                self.a_pkm[t_id].hp,
                self.p_pkm[t_id].p_type,
                self.p_pkm[t_id].hp,
                self.a_pkm[not t_id].p_type,
                self.a_pkm[not t_id].hp,
                self.a_pkm[t_id].moves[0].type, self.a_pkm[t_id].moves[0].power,
                self.a_pkm[t_id].moves[1].type, self.a_pkm[t_id].moves[1].power,
                self.a_pkm[t_id].moves[2].type, self.a_pkm[t_id].moves[2].power,
                self.a_pkm[t_id].moves[3].type, self.a_pkm[t_id].moves[3].power,
                self.p_pkm[t_id].moves[0].type, self.p_pkm[t_id].moves[0].power,
                self.p_pkm[t_id].moves[1].type, self.p_pkm[t_id].moves[1].power,
                self.p_pkm[t_id].moves[2].type, self.p_pkm[t_id].moves[2].power,
                self.p_pkm[t_id].moves[3].type, self.p_pkm[t_id].moves[3].power,
                self.p_pkm[not t_id].p_type,
                self.p_pkm[not t_id].hp]

    def _switch_pkm(self, t_id):
        if self.p_pkm[t_id].hp != 0:
            temp = self.a_pkm[t_id]
            self.a_pkm[t_id] = self.p_pkm[t_id]
            self.p_pkm[t_id] = temp
            if self.debug:
                self.debug_message[t_id] = "SWITCH"
            self.switched[t_id] = True
        elif self.debug:
            self.debug_message[t_id] = "FAILED SWITCH"

    def _attack_pkm(self, t_id, m_id):
        move = self.a_pkm[t_id].moves[m_id]
        opponent_pkm = self.a_pkm[not t_id]
        effectiveness_multiplier = calc_type_multiplier(move.type, opponent_pkm.p_type)
        stab_multiplier = 1.5 if move.type in self.a_pkm[t_id].p_type else 1.
        damage = move.power * effectiveness_multiplier * stab_multiplier
        opponent_pkm.hp = max(opponent_pkm.hp - damage, 0.)
        if self.debug:
            self.debug_message[t_id] = "ATTACK with " + str(move) + " to type " + TYPE_TO_STR[
                opponent_pkm.p_type[0]] + '/' + TYPE_TO_STR[opponent_pkm.p_type[1]] + " multiplier=" + str(
                effectiveness_multiplier) + " causing " + str(damage) + " damage, leaving opponent hp " + str(
                opponent_pkm.hp) + ''
        return damage
    
    def _battle_pkm(self, a, t_id):
        """
        Executes the battle

        :param a: attack
        :param t_id: trainer id
        :return: reward, terminal, and whether target survived and can attack
        """
        opponent = not t_id
        terminal = False
        next_player_can_attack = True
        damage_dealt = self._attack_pkm(t_id, a)
        reward = damage_dealt / HIT_POINTS
        if self._fainted_pkm(self.a_pkm[opponent]):
            self.has_fainted = True
            reward += 1.
            next_player_can_attack = False
            if self._fainted_pkm(self.p_pkm[opponent]):
                terminal = True
            else:
                self._switch_pkm(opponent)
                if self.debug:
                    self.debug_message[opponent] += " FAINTED"
        return reward, terminal, next_player_can_attack, damage_dealt
    
    @staticmethod
    def _fainted_pkm(pkm):
        return pkm.hp == 0
    
class SimpleMove:
    def __init__(self, move_type=None, move_power=None, my_type=None):
        if move_type is None or move_type == NONE:
            self.type = get_random_type(NONE)
        else:
            self.type = move_type
        if move_power is None:
            self.power = np.random.randint(POWER_MIN, POWER_MAX)
        else:
            self.power = move_power

    def __str__(self):
        return "Move(" + TYPE_TO_STR[self.type] + ", " + str(self.power) + ")"

class SimplePkm:
    def __init__(self, p_type=None, hp=HIT_POINTS,
                 type0=None, type0power=None, type1=None, type1power=None,
                 type2=None, type2power=None, type3=None, type3power=None):
        self.hp = hp
        if p_type is None:
            self.p_type = get_random_type_combo()
            self.moves = [SimpleMove(my_type=self.p_type) for i in range(N_MOVES - 1)] + [
                SimpleMove(move_type=self.p_type[0])]
        else:
            if isinstance(p_type, tuple):
                self.p_type = p_type
            else:
                self.p_type = (p_type, NONE)
            self.moves = [SimpleMove(move_type=type0, move_power=type0power),
                          SimpleMove(move_type=type1, move_power=type1power),
                          SimpleMove(move_type=type2, move_power=type2power),
                          SimpleMove(move_type=type3, move_power=type3power)]

    def __str__(self):
        return 'Pokemon(' + TYPE_TO_STR[self.p_type[0]] + '/' + TYPE_TO_STR[self.p_type[1]] + ', HP ' + str(self.hp) + ', ' + str(self.moves[0]) + ', ' + str(
            self.moves[1]) + ', ' + str(self.moves[2]) + ', ' + str(self.moves[3]) + ')'


def encode(s):
    """
    Encode Game state.
    
    :param s: game state
    :return: encoded game state in one hot vector
    """
    e = []
    for i in range(0, len(s)):
        s_i = s[i]
        if i % 2 == 0:
            # It's a type
            if not isinstance(s_i, tuple):
               s_i = [s_i]
            for t in s_i:
                b = [0] * N_TYPES
                b[t+1] = 1
                e.extend(b)
        else:
            # It's a value (hp or power)
            e.append(s_i / HIT_POINTS)
    return e

def decode(e):
    """
    Decode game state.

    :param e: encoded game state in one hot vector
    :return: game state
    """
    s = []
    index_e = 0
    for i in range(0, 7):
        for j in range(index_e, index_e + N_TYPES):
            if e[j] == 1:
                s.append(j % (N_TYPES + 1))
        index_e += N_TYPES
        s.append(e[index_e] * HIT_POINTS)
        index_e += 1
    s.append(e[index_e] * HIT_POINTS)
    return s


def get_super_effective_move(t):
    """
    :param t: pokemon type
    :return: a random type that is super effective against pokemon type t
    """
    s = [type for type in (TYPE_LIST[1:]) if 
         calc_type_multiplier(type, t) > 1.]
    if not s:
        print('Warning: Empty List!')
        return get_random_type(NONE)
    return np.random.choice(s)


def get_non_very_effective_move(t):
    """
    :param t: pokemon type
    :return: a random type that is not very effective against pokemon type t
    """
    s = [type for type in (TYPE_LIST[1:]) if 
         calc_type_multiplier(type, t) < 1.]
    if not s:
        print('Warning: Empty List!')
        return get_random_type(NONE)
    return np.random.choice(s)


def get_normal_effective_move(t):
    """
    :param t: pokemon type
    :return: a random type that is not very effective against pokemon type t
    """
    s = [type for type in (TYPE_LIST[1:]) if 
         calc_type_multiplier(type, t) == 1.]
    if not s:
        print('Warning: Empty List!')
        return get_random_type(NONE)
    return np.random.choice(s)

def get_random_type_combo(t=None):
    """
    Generates a random type combo.

    :param t: pokemon type.
    :return: a random type combo, with its primary type being super effective against the type t.
    """
    if t is None:
        primary = get_random_type(NONE)
    else:
        primary = get_super_effective_move(t)
    secondary = get_random_type(primary)
    return (primary, secondary)

def calc_type_multiplier(att_type, def_type):
    """
    Calculates the type advantage multiplier.

    :param att_type: the type of the attacking move.
    :param def_type: the type of the defending pokemon.
    :return: the type advantage multiplier.
    """
    if isinstance(def_type, tuple):
        mult_type1 = TYPE_CHART_MULTIPLIER[att_type][def_type[0]]
        mult_type2 = TYPE_CHART_MULTIPLIER[att_type][def_type[1]] if def_type[1] != NONE else 1.
        return mult_type1 * mult_type2
    return TYPE_CHART_MULTIPLIER[att_type][def_type]

def get_random_type(exclude=None):
    """
    :param exclude: the types that shouldn't be chosen.
    :return: a random pokemon type that isn't excluded.
    """
    if isinstance(exclude, (list, tuple)):
        return np.random.choice(TYPE_LIST[~np.isin(TYPE_LIST,exclude)])
    return np.random.choice(TYPE_LIST[TYPE_LIST != exclude])

def get_meta_aware_moves(t):
    """
    :param t: Type of the pokemon
    :return: Shuffled moves using simple meta strategies
    """
    moves = [
        t[0], 
        t[1], 
        get_coverage_move(t), 
        get_coverage_move(t)
    ]
    np.random.shuffle(moves)
    return moves

def get_coverage_move(t):
    """
    :param t: Type of the pokemon
    :return: Returns a type t'' that is super effective against a type t' the is super effective against the type t
    """
    return get_super_effective_move(get_super_effective_move(t))
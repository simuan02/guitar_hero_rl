import time

import gymnasium as gym
import numpy as np
import pygame

from guitar_hero.app.nota import Nota

"""Ambiente di gioco, in cui sono definiti: spazio degli stati, spazio delle azioni e logica di gioco"""


# Definizione dei Note Targets
def target_definition():
    note_targets_x_center = {
        "verde": 76,
        "rossa": 214,
        "gialla": 350,
        "blu": 486,
        "arancione": 624
    }
    note_targets_y_center = 637
    targets = {}
    for colore, x_pos in note_targets_x_center.items():
        targets[colore] = pygame.Rect(0, 0, 112, 80)
        targets[colore].center = (x_pos, note_targets_y_center)
    return targets


class GuitarHeroEnv(gym.Env):
    def __init__(self, collection, screen=None, clock=None, render_mode="human", discrete_state=False, model_testing=False):
        super(GuitarHeroEnv, self).__init__()

        # Definizione dello Spazio delle Possibili Azioni, corrispondenti con la
        # Pressione di uno dei Tasti delle 5 Note:
        # 0: Nessuna pressione; 1: Verde; 2: Rosso; 3: Giallo; 4: Blu; 5: Arancione
        self.action_space = gym.spaces.Discrete(6)

        # Definizione dello Spazio delle Osservazioni, corrispondenti alle posizioni
        # Delle 3 Note più vicine al fondo in ognuna delle 5 Corsie (Corsia Verde, Corsia Rossa, Corsia Gialla,
        # Corsia Blu, Corsia Arancione):
        # 0: Vuota; 1: Note Troppo Lontane; 2: Almeno una Nota Vicina (non in Posizione Target);
        # 3: Almeno una Nota in Posizione Target; 4: Note molto oltre il Target
        self.observation_space = gym.spaces.MultiDiscrete([5, 5, 5, 5, 5])

        self.collection = collection

        self.render_mode = render_mode

        self.discrete_state = discrete_state

        self.model_testing = model_testing

        self.reset()

        if self.render_mode == "human":
            self.screen = screen
            self.clock = clock

            self.text_font = pygame.font.SysFont("Segoe UI Black", 36)
            self.text_font_small = pygame.font.SysFont("Segoe UI Black", 24)
            self.active_color = None
            # Superficie esterna alla highway
            self.hit_light = (100, 245, 100, 100)  # Verde semi-trasparente
            self.miss_light = (245, 100, 100, 100)  # Rosso semi-trasparente
            self.partial_hit_light = (245, 245, 100, 100)  # Giallo semi-trasparente
            self.effect_timer = 0

            highway_image = pygame.image.load("images/highway.png").convert_alpha()
            self.highway_scaled = pygame.transform.scale(highway_image, (700, 750))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score_var = 0

        # Moltiplicatore di gioco limitato a 10, per evitare problemi di overflow in fase di calcolo del punteggio totale
        self.multiplier_var = 1

        self.note_clicking_mode_counter = {"Perfect": 0, "Imperfect": 0, "Missed": 0, "Misclick": 0}

        self.targets = target_definition()
        self.note_group = pygame.sprite.Group()

        self.song = self.collection.aggregate([{"$sample": {"size": 1}}]).next()

        if self.model_testing:
            self.song_notes = [
                nota for nota in self.song["note"] if nota["tempo"] <= 30_000
            ]
        else:
            self.song_notes = self.song["note"]

        # print(self.song["title"])

        self.lanes_state = {
            "verde": 0,
            "rossa": 0,
            "gialla": 0,
            "blu": 0,
            "arancione": 0
        }

        self.note_index = 0
        self.current_time = 0.0
        self.terminated = False

        self.last_step_time = time.perf_counter()

        self.stats_showed = False

        self.no_click_bonus = 0

        self.consecutive_misclicks = 0

        if self.discrete_state:
            return self._get_discrete_state_from_obs(self._get_obs()), {}
        else:
            return self._get_obs(), {}

    # Funzione che ottiene le osservazioni dall'ambiente, ossia gli stati delle corsie,
    # definiti in base alla posizione della nota più vicina al fondo:
    # 0: Nessuna nota presente in corsia;
    # 1: Nota lontana (La nota non ha superato il target e Distanza > 75);
    # 2: Nota vicina (Distanza dal target compresa tra 25 e 75);
    # 3: Nota nel target (Distanza dal target <= 25);
    # 4: Nota mancata (La nota ha superato il target e Distanza > 75);
    def _get_obs(self):
        colors = ["verde", "rossa", "gialla", "blu", "arancione"]
        for color in colors:
            lane_notes = [nota for nota in self.note_group if nota.colore_nota == color]
            # Se non ci sono note nella corsia, stato = 0
            if not lane_notes:
                self.lanes_state[color] = 0
            else:
                sorted_notes = sorted(lane_notes, key=lambda nota: nota.rect.y, reverse=True)
                nearest_note = sorted_notes[0]
                distance = nearest_note.rect.y - self.targets[color].y
                if distance < -75:
                    self.lanes_state[color] = 1
                elif 25 < abs(distance) <= 75:
                    self.lanes_state[color] = 2
                elif abs(distance) <= 25:
                    self.lanes_state[color] = 3
                elif distance > 75:
                    self.lanes_state[color] = 4

        return np.array([self.lanes_state[c] for c in colors], dtype=np.int64)

    def step(self, action):
        reward = 0
        color_map = {1: "verde",
                     2: "rossa",
                     3: "gialla",
                     4: "blu",
                     5: "arancione"}
        info = {}

        # Consideriamo un'azione che non sia la non pressione di un tasto
        if action > 0:
            reward = self.collision_control(color_map[action])
        else:
            clickable_lanes = [
                lane for lane in self.lanes_state.items() if lane[1] > 1
            ]

            if len(clickable_lanes) == 0:
                self.no_click_bonus = 0.1

        if self.render_mode == "human":
            current_real_time = time.perf_counter()
            dt = current_real_time - self.last_step_time
            self.last_step_time = current_real_time
        else:
            dt = 1/30.0

        note_group_before_update = [nota for nota in self.note_group]
        self.note_group.update(dt)

        if len(note_group_before_update) != len(self.note_group):
            reward -= 10 * (len(note_group_before_update) - len(self.note_group))
            self.score_var += reward
            self.multiplier_var = 1
            if self.render_mode == "human":
                self.active_color = self.miss_light
                self.effect_timer = 10
            difference = [item for item in note_group_before_update if item not in self.note_group]
            self.note_clicking_mode_counter["Missed"] += len(difference)
            self.display_score_and_multiplier()
            self.no_click_bonus = 0
        else:
            reward += self.no_click_bonus
            self.no_click_bonus = 0

        self.current_time += dt * 1000
        while self.note_index < len(self.song_notes) and \
                self.current_time >= self.song_notes[self.note_index]["tempo"]:
            self.note_group.add(Nota(self.song_notes[self.note_index], render_mode=self.render_mode))
            self.note_index += 1

        if self.note_index == len(self.song_notes) and not self.note_group:
            self.terminated = True
            if not self.stats_showed:
                info = self.show_execution_stats()
                self.stats_showed = True


        raw_obs = self._get_obs()

        if self.discrete_state:
            obs = self._get_discrete_state_from_obs(raw_obs)
        else:
            obs = raw_obs
        return obs, reward, self.terminated, False, info

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.blit(self.highway_scaled, (0, 0))
        self.note_group.draw(self.screen)
        self.display_score_and_multiplier()

        if self.active_color:
            self.highway_edge_color()

        pygame.display.flip()
        self.clock.tick(60)

    # Funzione per mostrare il punteggio e il moltiplicatore correnti
    def display_score_and_multiplier(self):
        if self.render_mode != "human":
            return
        score_surface = self.text_font.render(f"SCORE: {int(self.score_var)}", True, (255, 255, 255))
        score_rect = score_surface.get_rect(topright=(680, 20))
        glow_surface = self.text_font.render(f"SCORE: {int(self.score_var)}", True, (0, 255, 255))

        multiplier_surface = self.text_font_small.render(f"MULTIPLIER: x{self.multiplier_var}", True, (255, 255, 255))
        multiplier_rect = multiplier_surface.get_rect(topright=(660, 60))
        glow_surface_2 = self.text_font_small.render(f"MULTIPLIER: x{self.multiplier_var}", True, (0, 255, 255))

        self.screen.blit(glow_surface, score_rect.move(2, 2))
        self.screen.blit(score_surface, score_rect)
        self.screen.blit(glow_surface_2, multiplier_rect.move(2, 2))
        self.screen.blit(multiplier_surface, multiplier_rect)

    def collision_control(self, color):
        for nota1 in self.note_group:
            if nota1.colore_nota == color:
                if nota1.rect.colliderect(self.targets[color]):
                    y_difference = nota1.rect.y - self.targets[color].y
                    reward = 0
                    if 50 < abs(y_difference) < 75:
                        reward = 1 * self.multiplier_var
                        if self.render_mode == "human":
                            self.active_color = self.partial_hit_light
                            self.effect_timer = 10
                        self.note_clicking_mode_counter["Imperfect"] += 1
                    elif 25 < abs(y_difference) <= 50:
                        reward = 2 * self.multiplier_var
                        if self.render_mode == "human":
                            self.active_color = self.partial_hit_light
                            self.effect_timer = 10
                        self.note_clicking_mode_counter["Imperfect"] += 1
                    elif abs(y_difference) <= 25:
                        reward = 5 * self.multiplier_var
                        if self.render_mode == "human":
                            self.active_color = self.hit_light
                            self.effect_timer = 10
                        if self.multiplier_var < 10:
                            self.multiplier_var += 1
                        self.note_clicking_mode_counter["Perfect"] += 1
                    if abs(y_difference) >= 75:
                        reward = -1 - (0.5 * self.consecutive_misclicks)
                        self.consecutive_misclicks += 1
                        self.score_var -= 1
                        self.multiplier_var = 1
                        if self.render_mode == "human":
                            self.active_color = self.miss_light
                            self.effect_timer = 10
                        self.note_clicking_mode_counter["Misclick"] += 1
                        self.display_score_and_multiplier()
                        return reward
                    nota1.kill()
                    self.score_var += reward
                    if self.render_mode == "human":
                        self.display_score_and_multiplier()
                    self.consecutive_misclicks = 0
                    return reward
        self.score_var -= 5
        self.multiplier_var = 1
        if self.render_mode == "human":
            self.active_color = self.miss_light
            self.effect_timer = 10
        self.note_clicking_mode_counter["Misclick"] += 1
        if self.render_mode == "human":
            self.display_score_and_multiplier()
        reward = -5 - (0.5 * self.consecutive_misclicks)
        self.consecutive_misclicks += 1
        return reward

    # Funzione per la colorazione della parte laterale alla highway in caso di pressione di un tasto
    # Corrispondente a una nota
    def highway_edge_color(self):
        if self.effect_timer > 0:
            # Creiamo una superficie con supporto per la trasparenza
            overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)

            # Disegnamo due poligoni ai lati della highway
            left_polygon_points = [(0, 650), (0, 271), (233, 246), (292, 251)]
            pygame.draw.polygon(overlay, self.active_color, left_polygon_points)

            right_polygon_points = [(700, 645), (406, 251), (469, 245), (700, 267)]
            pygame.draw.polygon(overlay, self.active_color, right_polygon_points)

            self.screen.blit(overlay, (0, 0))
            self.effect_timer -= 1
        else:
            self.active_color = None

    def show_execution_stats(self):
        max_possible_score = 0
        for i in range(1, len(self.song_notes) + 1):
            max_possible_score += 5 * i if i < 10 else 50
        print("Score: ", int(self.score_var), " - Maximum Possible Score: ", max_possible_score)
        print("Perfectly Clicked Notes: ", self.note_clicking_mode_counter["Perfect"])
        print("Imperfectly Clicked Notes: ", self.note_clicking_mode_counter["Imperfect"])
        print("Missed Notes: ", self.note_clicking_mode_counter["Missed"])
        print("Misclicks:", self.note_clicking_mode_counter["Misclick"])
        clicks = self.note_clicking_mode_counter["Perfect"] + self.note_clicking_mode_counter["Imperfect"] + \
                 self.note_clicking_mode_counter["Misclick"]
        perfect_precision = round((self.note_clicking_mode_counter["Perfect"] / max(1, clicks)) * 100, 2)
        precision = round(
            ((self.note_clicking_mode_counter["Perfect"] + self.note_clicking_mode_counter["Imperfect"]) / max(1, clicks)) * 100, 2)
        print("Precision: ", perfect_precision, "%")
        print("Precision (with imperfect clicks): ", precision, "%")
        perfect_recall = round((self.note_clicking_mode_counter["Perfect"] / len(self.song_notes)) * 100, 2)
        print("Recall: ", perfect_recall, "%")
        recall = round(((self.note_clicking_mode_counter["Perfect"] + self.note_clicking_mode_counter["Imperfect"]) /
                        len(self.song_notes)) * 100, 2)
        print("Recall (with imperfect clicks): ", recall, "%")
        p = perfect_precision / 100
        r = perfect_recall / 100
        f1_score = round((2 * p * r / (p + r)) * 100, 2) if p + r > 0 else 0
        print("F1-Score: ", f1_score, "%")
        info = {"precision": perfect_precision, "recall": perfect_recall, "f1_score": f1_score}
        return info

    # Funzione per ottenere lo stato in forma discreta, che restituisce una coppia lane_mask, una 5-upla contenente
    # un valore intero rappresentante lo stato ottenuto dall'osservazione per ogni corsia
    def _get_discrete_state_from_obs(self, obs):
        lane_mask = tuple(int(lane) for lane in obs)

        return lane_mask

    # Funzione per ottenere lo stato attuale in formato testuale, da passare al LLM
    def get_llm_state_description(self, obs):
        lanes = ["verde", "rossa", "gialla", "blu", "arancione"]
        status_map = {
            0: "vuota",
            1: "nota lontana",
            2: "nota in avvicinamento",
            3: "NOTA NEL TARGET",
            4: "nota mancata"
        }

        desc_parts = []
        empty_highway = True

        for i, val in enumerate(obs):
            if val > 0:
                empty_highway = False
            desc_parts.append(f"{lanes[i]}: {status_map[val]}")

        if empty_highway:
            return "Tutte le corsie sono vuote. Azione consigliata: 0."

        return " | ".join(desc_parts)

    # Funzione per ottenere una descrizione testuale dello stato corrente e del risultato di un'azione
    def get_outcome_description(self, obs, action):
        lanes = ["v", "r", "g", "b", "a"]
        status_map = {0: "empty", 1: "far", 2: "near",
                      3: "TARGET", 4: "missed"}
        click_action_result = {0: "Missed", 1: "Imperfect", 2: "Perfect"}

        state_desc = " | ".join([f"{lanes[i]}: {status_map[obs[i]]}" for i in range(5)])

        note_outcome = []
        for i in range(5):
            if action == 0:
                if obs[i] == 3:
                    note_outcome.append(f"{lanes[i]}: Missed")
                else:
                    note_outcome.append(f"{lanes[i]}: No_Action")
            else:
                if i + 1 == action:
                    if obs[i] == 0 or obs[i] == 1 or obs[i] == 4:
                        result_index = 0
                    elif obs[i] == 2:
                        result_index = 1
                    else:
                        result_index = 2
                    note_outcome.append(f"{lanes[i]}: {click_action_result[result_index]}")
                else:
                    note_outcome.append(f"{lanes[i]}: not_targeted")

        if action == 0:
            action_desc = "no_click"
        else:
            action_desc = f"click_{lanes[action - 1]}"

        outcome_desc = (
                f"STA:\n{state_desc}\n\n"
                f"ACT:\n{action_desc}\n\n"
                f"OUT:\n- " + "\n- ".join(note_outcome) + "\n"
                f"MUL:\n{self.multiplier_var}\n"
                f"ERR:\n{self.consecutive_misclicks}\n"
        )

        return outcome_desc

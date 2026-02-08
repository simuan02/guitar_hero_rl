import sys
import tempfile

import gridfs
import pygame
import gymnasium as gym
from pymongo import MongoClient

from nota import Nota


# Funzione per il controllo delle collisioni delle note con i target, che è invocata quando c'è la pressione di uno
# dei 5 tasti previsti (1 - 2 - 3 - 4 - 5)
def collision_control(color):
    global score_var, multiplier_var, active_color
    print(lanes_state)
    for nota1 in note_group:
        if nota1.colore_nota == color:
            if nota1.rect.colliderect(targets[color]):
                print("Note Position: (", nota1.rect.x, ", ", nota1.rect.y, ")")
                y_difference = nota1.rect.y - targets[color].y
                if 50 < abs(y_difference) < 75:
                    score_var += 1 * multiplier_var
                    active_color = partial_hit_light
                    print(
                        "Input: Nota " + color + "; Decisione: premuta parzialmente; Variazione Punteggio: +1 * " + str(
                            multiplier_var) +
                        "; Variazione Moltiplicatore: Invariato")
                    note_clicking_mode_counter["Imperfect"] += 1
                elif 25 < abs(y_difference) <= 50:
                    score_var += 2 * multiplier_var
                    active_color = partial_hit_light
                    print(
                        "Input: Nota " + color + "; Decisione: premuta quasi correttamente; Variazione Punteggio: +2 * "
                        + str(multiplier_var) + "; Variazione Moltiplicatore: Invariato")
                    note_clicking_mode_counter["Imperfect"] += 1
                elif abs(y_difference) <= 25:
                    score_var += 5 * multiplier_var
                    active_color = hit_light
                    multiplier_var += 1
                    print(
                        "Input: Nota " + color + "; Decisione: premuta perfettamente; Variazione Punteggio: +5 * " + str(
                            multiplier_var) +
                        "; Variazione Moltiplicatore: +1")
                    note_clicking_mode_counter["Perfect"] += 1
                if abs(y_difference) >= 75:
                    print("Input: Nota " + color + "; Decisione: mancata; Variazione Punteggio: -1" +
                          "; Variazione Moltiplicatore: ritorna a 1")
                    score_var -= 1
                    multiplier_var = 1
                    active_color = miss_light
                    note_clicking_mode_counter["Missed"] += 1
                    display_score_and_multiplier()
                    return False
                nota1.kill()
                display_score_and_multiplier()
                return True
    print("Input: Nota " + color + "; Decisione: mancata o assente; Variazione Punteggio: -2" +
          "; Variazione Moltiplicatore: ritorna a 1")
    score_var -= 2
    multiplier_var = 1
    active_color = miss_light
    note_clicking_mode_counter["Missed"] += 1
    display_score_and_multiplier()
    return False


# Funzione per mostrare il punteggio e il moltiplicatore correnti
def display_score_and_multiplier():
    score_surface = text_font.render(f"SCORE: {score_var}", True, (255, 255, 255))
    score_rect = score_surface.get_rect(topright=(680, 20))
    glow_surface = text_font.render(f"SCORE: {score_var}", True, (0, 255, 255))

    multiplier_surface = text_font_small.render(f"MULTIPLIER: x{multiplier_var}", True, (255, 255, 255))
    multiplier_rect = score_surface.get_rect(topright=(660, 60))
    glow_surface_2 = text_font_small.render(f"MULTIPLIER: x{multiplier_var}", True, (0, 255, 255))

    screen.blit(glow_surface, score_rect.move(2, 2))
    screen.blit(score_surface, score_rect)
    screen.blit(glow_surface_2, multiplier_rect.move(2, 2))
    screen.blit(multiplier_surface, multiplier_rect)


# Funzione per la colorazione della parte laterale alla highway in caso di pressione di un tasto
# Corrispondente a una nota
def highway_edge_color(color):
    global effect_timer, active_color
    if effect_timer > 0:
        # Creiamo una superficie con supporto per la trasparenza
        overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)

        # Disegnamo due poligoni ai lati della highway
        left_polygon_points = [(0, 650), (0, 271), (233, 246), (292, 251)]
        pygame.draw.polygon(overlay, color, left_polygon_points)

        right_polygon_points = [(700, 645), (406, 251), (469, 245), (700, 267)]
        pygame.draw.polygon(overlay, color, right_polygon_points)

        screen.blit(overlay, (0, 0))
        effect_timer -= 1
    else:
        active_color = None


# Funzione per ottenere una canzone a caso dal DB
def get_random_song():
    return collection.aggregate([{"$sample": {"size": 1}}]).next()


# Funzione per la definzione degli stati delle corsie, definiti in base alla posizione della nota più vicina al fondo:
# 0: Nessuna nota presente in corsia;
# 1: Nota lontana (La nota non ha superato il target e Distanza > 75);
# 2: Nota vicina (Distanza dal target compresa tra 25 e 75);
# 3: Nota nel target (Distanza dal target <= 25);
# 4: Nota mancata (La nota ha superato il target e Distanza > 75);
def update_lanes_state():
    for color in ["verde", "rossa", "gialla", "blu", "arancione"]:
        lane_notes = [nota for nota in note_group if nota.colore_nota == color]
        # Se non ci sono note nella corsia, stato = 0
        if not lane_notes:
            lanes_state[color] = 0
        else:
            nearest_to_baseline_note = max(lane_notes, key=lambda nota: nota.rect.y)
            distance = nearest_to_baseline_note.rect.y - targets[color].y
            if distance < -75:
                lanes_state[color] = 1
            elif 25 < abs(distance) <= 75:
                lanes_state[color] = 2
            elif abs(distance) <= 25:
                lanes_state[color] = 3
            elif distance > 75:
                lanes_state[color] = 4


# Connessione DB
client = MongoClient("mongodb://localhost:27017/")
db = client["songs"]
collection = db["songs"]

fs = gridfs.GridFS(db)
song = None

pygame.init()
screen = pygame.display.set_mode((700, 750))
pygame.display.set_caption("Guitar Hero", "Guitar Hero")
clock = pygame.time.Clock()

background_image = pygame.image.load("images/background.png").convert_alpha()
background_scaled = pygame.transform.scale(background_image, (700, 750))

background2_image = pygame.image.load("images/background2.png").convert_alpha()
background2_scaled = pygame.transform.scale(background2_image, (700, 750))

highway_image = pygame.image.load("images/highway.png").convert_alpha()
highway_scaled = pygame.transform.scale(highway_image, (700, 750))

game_state = "MENU"
first = True

title_font = pygame.font.SysFont("Impact", 72, bold=False)
text_font = pygame.font.SysFont("Segoe UI Black", 36, bold=False)
text_font_small = pygame.font.SysFont("Segoe UI Black", 24, bold=False)
text_font_mini = pygame.font.SysFont("Segoe UI Black", 12, bold=False)
text_font_big = pygame.font.SysFont("Segoe UI Black", 52, bold=False)

title_surface = title_font.render("GUITAR HERO", False, "white")
title_surface_rect = title_surface.get_rect(center=(350, 60))

start_rect = pygame.Rect(200, 350, 300, 50)
start_text_surface = text_font.render("START GAME", True, (255, 255, 255))
start_text_rect = start_text_surface.get_rect(center=start_rect.center)
start_rect2 = pygame.Rect(200, 375, 300, 50)
start_text_rect2 = start_text_surface.get_rect(center=start_rect2.center)

# Rules
rules_rect = pygame.Rect(200, 500, 300, 50)
rules_text_surface = text_font.render("GAME RULES", True, (255, 255, 255))
rules_text_rect = rules_text_surface.get_rect(center=rules_rect.center)
rules = ["Guitar Hero is a musical game.",
         "Songs are represented by a set of 5 possible notes: Green, Red, Yellow, Blue, Orange.",
         "The goal of the game is to maximise the score, ",
         "i.e. the sum of single scores calculated as following described: ",
         "5 points if the note is perfectly timed clicked;",
         "2 points if the note is almost perfectly timed clicked;",
         "1 point if the note is clicked in an acceptable time;",
         "-1 point if the note is not clicked in time;",
         "-2 point if the note is missed or in case of a misclick.",
         "The multiplier starts from 1 and resets in case of negative single score."]
rules_page_rect = pygame.Rect(50, 100, 600, 500)
rules_y = rules_page_rect.top
row_height = text_font_small.get_linesize()
rules_page_rects = []
for i, row in enumerate(rules):
    # Genera la superficie per la singola riga
    sentence = text_font_mini.render(row, True, (255, 255, 255))
    sentence_rect = pygame.Rect(50, rules_y, 600, 50)
    sentence_text_rect = sentence.get_rect(left=sentence_rect.left, center=sentence_rect.center)
    rules_page_rects.append({"surface": sentence, "rect": sentence_rect, "text_rect": sentence_text_rect})
    rules_y += row_height
back_rect = pygame.Rect(150, 500, 400, 50)
back_surface = text_font.render("BACK TO THE GAME", True, (255, 255, 255))
back_text_rect = back_surface.get_rect(center=back_rect.center)

# Score
score_var = 0
final_score_rect = pygame.Rect(200, 200, 300, 50)
final_score_surface = text_font_big.render("SCORE: " + str(score_var), True, (255, 255, 255))
final_score_text_rect = final_score_surface.get_rect(center=final_score_rect.center)

# Moltiplicatore
multiplier_var = 1

# Conteggio Note
note_clicking_mode_counter = {"Perfect": 0, "Imperfect": 0, "Missed": 0}

# Note
note_group = pygame.sprite.Group()
note_timer = pygame.USEREVENT + 1
note_index = 0
note_list = []

# Note Targets
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

# Superficie esterna alla highway
hit_light = (100, 245, 100, 100)  # Verde semi-trasparente
miss_light = (245, 100, 100, 100)  # Rosso semi-trasparente
partial_hit_light = (245, 245, 100, 100)  # Giallo semi-trasparente
effect_timer = 0
active_color = None

note_time = 0

end_game_timer = pygame.USEREVENT + 2
end_game_timer_set = False

# Stato delle corsie
lanes_state = {
    "verde": 0,
    "rossa": 0,
    "gialla": 0,
    "blu": 0,
    "arancione": 0
}

while True:

    dt = clock.tick(60) / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        if game_state == "ACTIVE":
            if event.type == end_game_timer:
                max_possible_score = 0
                for i in range(1, len(note_list) + 1):
                    max_possible_score += 5 * i
                print("Score: ", score_var, " - Maximum Possible Score: ", max_possible_score)
                print("Perfectly Clicked Notes: ", note_clicking_mode_counter["Perfect"])
                print("Imperfectly Clicked Notes: ", note_clicking_mode_counter["Imperfect"])
                print("Missed Clicks: ", note_clicking_mode_counter["Missed"])
                clicks = note_clicking_mode_counter["Perfect"] + note_clicking_mode_counter["Imperfect"] + note_clicking_mode_counter["Missed"]
                precision = round((note_clicking_mode_counter["Perfect"] / clicks) * 100, 2)
                imperfect_precision = round(((note_clicking_mode_counter["Perfect"] + note_clicking_mode_counter["Imperfect"]) / clicks) * 100, 2)
                print("Precision: ", precision, "%")
                print("Precision (with imperfect clicks): 1", imperfect_precision, "%")
                note_clicking_mode_counter["Perfect"] = note_clicking_mode_counter["Imperfect"] = (
                    note_clicking_mode_counter)["Missed"] = 0
                lanes_state = {
                    "verde": 0,
                    "rossa": 0,
                    "gialla": 0,
                    "blu": 0,
                    "arancione": 0
                }
                game_state = "MENU"
                note_index = 0
                multiplier_var = 1
                note_group.empty()
                pygame.time.set_timer(end_game_timer, -1)
                note_time = 0
                end_game_timer_set = False
                pygame.mixer.music.stop()

            if event.type == pygame.KEYDOWN:
                hit_success = None
                if event.key == pygame.K_1:
                    hit_success = collision_control("verde")
                if event.key == pygame.K_2:
                    hit_success = collision_control("rossa")
                if event.key == pygame.K_3:
                    hit_success = collision_control("gialla")
                if event.key == pygame.K_4:
                    hit_success = collision_control("blu")
                if event.key == pygame.K_5:
                    hit_success = collision_control("arancione")
                if hit_success is not None:
                    effect_timer = 15

        else:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not game_state == "RULES":
                    if start_rect.collidepoint(event.pos) or start_rect2.collidepoint(event.pos):
                        score_var = 0
                        game_state = "ACTIVE"
                        if first:
                            first = False
                        song = get_random_song()
                        print(str(song))
                        note_list = song["note"]
                        note_list_len = len(note_list)
                        note_time = note_list[0]["tempo"]
                        pygame.time.set_timer(note_timer, note_list[0]["tempo"])
                        audio_id = song["audio_file_id"]
                        if audio_id:
                            audio_file = fs.get(audio_id)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio:
                                temp_audio.write(audio_file.read())
                                temp_path = temp_audio.name
                            pygame.mixer.music.load(temp_path)
                            pygame.mixer.music.play()
                        print(song["title"])
                    if rules_rect.collidepoint(event.pos):
                        game_state = "RULES"
                else:
                    if back_rect.collidepoint(event.pos):
                        game_state = "MENU"

    if game_state == "ACTIVE":
        screen.blit(highway_scaled, (0, 0))
        note_group.draw(screen)
        note_group_before_update = [nota for nota in note_group]
        note_group.update(dt)
        if len(note_group_before_update) != len(note_group):
            score_var -= 2 * (len(note_group_before_update) - len(note_group))
            multiplier_var = 1
            active_color = miss_light
            effect_timer = 15
            difference = [item for item in note_group_before_update if item not in note_group]
            for nota in difference:
                print("Input: Nota " + nota.colore_nota + "; Decisione: mancata; Variazione Punteggio: -2" +
                      "; Variazione Moltiplicatore: ritorna a 1")
                note_clicking_mode_counter["Missed"] += 1
        if active_color:
            highway_edge_color(active_color)
        display_score_and_multiplier()
        update_lanes_state()
        current_time = pygame.mixer.music.get_pos()
        while note_index < len(note_list) and current_time >= note_list[note_index]["tempo"]:
            note_group.add(Nota(note_list[note_index], render_mode="human"))
            note_index += 1
        if note_index == len(note_list) and not end_game_timer_set:
            pygame.time.set_timer(end_game_timer, int(note_list[note_index - 1]["tempo_percorrenza"] * 1000) + 2000)
            end_game_timer_set = True

    elif not game_state == "RULES":
        if not first:
            screen.blit(background2_scaled, (0, 0))
            screen.blit(title_surface, title_surface_rect)
            final_score_surface = text_font_big.render("SCORE: " + str(score_var), True, (255, 255, 255))
            final_score_text_rect = final_score_surface.get_rect(center=final_score_rect.center)
            pygame.draw.rect(screen, "black", start_rect2, border_radius=10)
            pygame.draw.rect(screen, "white", start_rect2, width=4, border_radius=10)
            screen.blit(start_text_surface, start_text_rect2)
            screen.blit(final_score_surface, final_score_text_rect)
        else:
            screen.blit(background_scaled, (0, 0))
            screen.blit(title_surface, title_surface_rect)
            pygame.draw.rect(screen, "black", start_rect, border_radius=10)
            pygame.draw.rect(screen, "white", start_rect, width=4, border_radius=10)
            screen.blit(start_text_surface, start_text_rect)
        pygame.draw.rect(screen, "black", rules_rect, border_radius=10)
        pygame.draw.rect(screen, "white", rules_rect, width=4, border_radius=10)
        screen.blit(rules_text_surface, rules_text_rect)
    else:
        screen.fill("black")
        screen.blit(title_surface, title_surface_rect)
        for sentence in rules_page_rects:
            screen.blit(sentence["surface"], sentence["text_rect"])
        pygame.draw.rect(screen, "black", back_rect, border_radius=10)
        pygame.draw.rect(screen, "white", back_rect, width=4, border_radius=10)
        screen.blit(back_surface, back_text_rect)

    pygame.display.update()

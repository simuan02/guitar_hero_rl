import tempfile

import gridfs
import pygame
from pymongo import MongoClient

from custom_env import GuitarHeroEnv

""" Questo script consente l'esecuzione dell'interfaccia utente completa, compresa di menu e schermata delle 
regole, che fornisce all'utente la possibilità di giocare"""

# Funzione per ottenere una canzone a caso dal DB
def get_random_song():
    # return collection.find_one({"title": "Seven Nation Army 2"})
    return collection.aggregate([{"$sample": {"size": 1}}]).next()


def handle_game_input(event):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_1:
            return 1
        if event.key == pygame.K_2:
            return 2
        if event.key == pygame.K_3:
            return 3
        if event.key == pygame.K_4:
            return 4
        if event.key == pygame.K_5:
            return 5
    return 0


class GameApp:
    def __init__(self):
        # Uno dei possibili stati di gioco: MENU (schermata di visualizzazione del menu di gioco
        # (prima schermata visualizzata all'avvio del gioco)) | RULES (schermata di
        # visualizzazione delle regole di gioco) | ACTIVE (interfaccia effettiva di gioco) | GAME_OVER (schermata successiva
        # alla terminazione di una partita)
        self.state = "MENU"
        self.env = None

        pygame.init()
        self.screen = pygame.display.set_mode((700, 750))
        pygame.display.set_caption("Guitar Hero")
        self.clock = pygame.time.Clock()

        self.active_color = None
        # Superficie esterna alla highway
        self.hit_light = (100, 245, 100, 100)  # Verde semi-trasparente
        self.miss_light = (245, 100, 100, 100)  # Rosso semi-trasparente
        self.partial_hit_light = (245, 245, 100, 100)  # Giallo semi-trasparente

        # Fonts
        self.title_font = pygame.font.SysFont("Impact", 72, bold=False)
        self.text_font = pygame.font.SysFont("Segoe UI Black", 36, bold=False)
        self.text_font_small = pygame.font.SysFont("Segoe UI Black", 24, bold=False)
        self.text_font_mini = pygame.font.SysFont("Segoe UI Black", 12, bold=False)
        self.text_font_big = pygame.font.SysFont("Segoe UI Black", 52, bold=False)

        # Background Components
        background_image = pygame.image.load("images/background.png").convert_alpha()
        self.background_scaled = pygame.transform.scale(background_image, (700, 750))

        self.title_surface = self.title_font.render("GUITAR HERO", False, "white")
        self.title_surface_rect = self.title_surface.get_rect(center=(350, 60))

        self.start_rect = pygame.Rect(200, 350, 300, 50)
        self.start_text_surface = self.text_font.render("START GAME", True, (255, 255, 255))
        self.start_text_rect = self.start_text_surface.get_rect(center=self.start_rect.center)
        self.start_rect2 = pygame.Rect(200, 375, 300, 50)
        self.start_text_rect2 = self.start_text_surface.get_rect(center=self.start_rect2.center)

        background2_image = pygame.image.load("images/background2.png").convert_alpha()
        self.background2_scaled = pygame.transform.scale(background2_image, (700, 750))

        highway_image = pygame.image.load("images/highway.png").convert_alpha()
        self.highway_scaled = pygame.transform.scale(highway_image, (700, 750))

        self.final_score_rect = pygame.Rect(200, 200, 300, 50)
        self.final_score_surface = self.text_font_big.render("SCORE: 0", True, (255, 255, 255))
        self.final_score_text_rect = self.final_score_surface.get_rect(center=self.final_score_rect.center)

        self.first = True
        self.end_game_timer = pygame.USEREVENT + 2

        self.rules_rect = pygame.Rect(200, 500, 300, 50)
        self.rules_text_surface = self.text_font.render("GAME RULES", True, (255, 255, 255))
        self.rules_text_rect = self.rules_text_surface.get_rect(center=self.rules_rect.center)
        rules = ["Guitar Hero is a musical game.",
                 "Songs are represented by a set of 5 possible notes: Green, Red, Yellow, Blue, Orange.",
                 "The goal of the game is to maximise the score, ",
                 "i.e. the sum of single scores calculated as following described: ",
                 "5 points if the note is perfectly timed clicked;",
                 "2 points if the note is almost perfectly timed clicked;",
                 "1 point if the note is clicked in an acceptable time;",
                 "-1 point if the note is not clicked in time;",
                 "-5 points if there is a misclick;",
                 "-10 points if the note have been missed.",
                 "The multiplier starts from 1 and resets in case of a negative single score."]
        self.rules_page_rect = pygame.Rect(50, 100, 600, 500)
        rules_y = self.rules_page_rect.top
        row_height = self.text_font_small.get_linesize()
        self.rules_page_rects = []
        for i, row in enumerate(rules):
            # Genera la superficie per la singola riga
            sentence = self.text_font_mini.render(row, True, (255, 255, 255))
            sentence_rect = pygame.Rect(50, rules_y, 600, 50)
            sentence_text_rect = sentence.get_rect(left=sentence_rect.left, center=sentence_rect.center)
            self.rules_page_rects.append({"surface": sentence, "rect": sentence_rect, "text_rect": sentence_text_rect})
            rules_y += row_height
        self.back_rect = pygame.Rect(150, 500, 400, 50)
        self.back_surface = self.text_font.render("BACK TO THE GAME", True, (255, 255, 255))
        self.back_text_rect = self.back_surface.get_rect(center=self.back_rect.center)

    # Funzione per la gestione dell'esecuzione del gioco in modalità umana (visualizzazione dell'interfaccia)
    def run(self):
        running = True
        end_game_timer_set = False

        while running:
            dt = self.clock.tick(60) / 1000.0
            action = 0

            for event in pygame.event.get():
                if event.type == self.end_game_timer:
                    pygame.time.set_timer(self.end_game_timer, -1)
                    end_game_timer_set = False
                    self.state = "GAME_OVER"
                    pygame.mixer.music.stop()

                if event.type == pygame.QUIT:
                    running = False

                if self.state == "MENU":
                    self.handle_menu_input(event)

                elif self.state == "RULES":
                    self.handle_rules_input(event)

                elif self.state == "ACTIVE":
                    res = handle_game_input(event)
                    if res > 0:
                        action = res
                elif self.state == "GAME_OVER":
                    self.handle_menu_input(event)

            if self.state == "ACTIVE":
                obs, reward, terminated, _, _ = self.env.step(action)

                if terminated and not end_game_timer_set:
                    pygame.time.set_timer(self.end_game_timer, 1000)
                    end_game_timer_set = True

            self.render()

            pygame.display.update()

    # Rendering dell'interfaccia di gioco, in base allo stato
    def render(self):
        if self.state == "MENU":
            self.draw_menu()

        elif self.state == "RULES":
            self.draw_rules()

        elif self.state == "ACTIVE":
            self.env.render()

        elif self.state == "GAME_OVER":
            self.draw_game_over()

    def draw_menu(self):
        self.screen.blit(self.background_scaled, (0, 0))
        self.screen.blit(self.title_surface, self.title_surface_rect)
        pygame.draw.rect(self.screen, "black", self.start_rect, border_radius=10)
        pygame.draw.rect(self.screen, "white", self.start_rect, width=4, border_radius=10)
        self.screen.blit(self.start_text_surface, self.start_text_rect)

        pygame.draw.rect(self.screen, "black", self.rules_rect, border_radius=10)
        pygame.draw.rect(self.screen, "white", self.rules_rect, width=4, border_radius=10)
        self.screen.blit(self.rules_text_surface, self.rules_text_rect)

    def draw_game_over(self):
        self.screen.blit(self.background2_scaled, (0, 0))
        self.screen.blit(self.title_surface, self.title_surface_rect)
        final_score_surface = self.text_font_big.render("SCORE: " + str(int(self.env.score_var)), True, (255, 255, 255))
        final_score_text_rect = final_score_surface.get_rect(center=self.final_score_rect.center)
        pygame.draw.rect(self.screen, "black", self.start_rect2, border_radius=10)
        pygame.draw.rect(self.screen, "white", self.start_rect2, width=4, border_radius=10)
        self.screen.blit(self.start_text_surface, self.start_text_rect2)
        self.screen.blit(final_score_surface, final_score_text_rect)

        pygame.draw.rect(self.screen, "black", self.rules_rect, border_radius=10)
        pygame.draw.rect(self.screen, "white", self.rules_rect, width=4, border_radius=10)
        self.screen.blit(self.rules_text_surface, self.rules_text_rect)

    def draw_rules(self):
        self.screen.fill("black")
        self.screen.blit(self.title_surface, self.title_surface_rect)
        for sentence in self.rules_page_rects:
            self.screen.blit(sentence["surface"], sentence["text_rect"])
        pygame.draw.rect(self.screen, "black", self.back_rect, border_radius=10)
        pygame.draw.rect(self.screen, "white", self.back_rect, width=4, border_radius=10)
        self.screen.blit(self.back_surface, self.back_text_rect)

    def handle_menu_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.start_rect.collidepoint(event.pos) or self.start_rect2.collidepoint(event.pos):
                self.env = GuitarHeroEnv(collection, self.screen, self.clock, render_mode="human")  # Caricamento dell'ambiente di gioco
                print("Start Game")
                self.state = "ACTIVE"
                audio_id = self.env.song["audio_file_id"]
                if audio_id:
                    audio_file = fs.get(audio_id)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio:
                        temp_audio.write(audio_file.read())
                        temp_path = temp_audio.name
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                print(self.env.song["title"])
            if self.rules_rect.collidepoint(event.pos):
                print("Show Rules")
                self.state = "RULES"

    def handle_rules_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_rect.collidepoint(event.pos):
                self.state = "MENU"


# Connessione DB
client = MongoClient("mongodb://localhost:27017/")
db = client["songs"]
collection = db["songs"]

fs = gridfs.GridFS(db)

game = GameApp()
game.run()

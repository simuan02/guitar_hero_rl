from random import randint

import pygame

"""
    Definizione della classe Sprite Nota, contenente informazioni quali colore e posizione corrente 
    all'interno dell'ambiente di gioco
"""

# Le corsie sono viste come quadrilateri e descritte attraverso le
# Coordinate dei quattro estremi (TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT)
lanes = {
    "verde": [(295, 252), (315, 252), (-50, 750), (97, 750)],
    "rossa": [(317, 252), (337, 252), (118, 750), (259, 750)],
    "gialla": [(339, 252), (359, 252), (279, 750), (422, 750)],
    "blu": [(362, 252), (381, 252), (444, 750), (582, 750)],
    "arancione": [(385, 252), (404, 252), (604, 750), (760, 750)]
}

note_targets_x_center = {
    "verde": 76,
    "rossa": 214,
    "gialla": 350,
    "blu": 486,
    "arancione": 624
}


def lane_interpolation(a, b, t):
    return a + (b - a) * t


class Nota(pygame.sprite.Sprite):
    def __init__(self, nota, render_mode=None):
        super().__init__()
        note = {
            "verde": "images/note/verde.png",
            "rossa": "images/note/rossa.png",
            "gialla": "images/note/gialla.png",
            "blu": "images/note/blu.png",
            "arancione": "images/note/arancione.png",
        }

        colore = nota["colore"]

        self.render_mode = render_mode

        if colore in note:
            path = note[colore]
            top_pos = (lanes[colore][0][0] + lanes[colore][1][0]) / 2
            if render_mode == "human":
                self.original_image = pygame.image.load(path).convert_alpha()
        else:
            raise ValueError(f"Nota sconosciuta: {colore}")

        self.colore_nota = colore

        if self.render_mode == "human":
            # La nota esiste fisicamente, quindi c'è bisogno di disegnarne l'immagine
            self.image = pygame.transform.scale(self.original_image, (40, 40))
            self.rect = self.image.get_rect(midtop=(top_pos, 252))
        else:
            # La nota esiste solo logicamente, ma il rettangolo è necessario per le osservazioni dell'ambiente
            self.image = None  # Non serve immagine all'AI
            self.rect = pygame.Rect(0, 0, 40, 40)
            self.rect.midtop = (int(top_pos), 252)

        # t è la variabile utilizzata per il movimento della nota nella rispettiva corsia
        self.t = 0.0

        self.tempo_percorrenza = nota["tempo_percorrenza"]

        self.hit = False

        self.update_size()

    def update(self, dt):
        self.t += (1/self.tempo_percorrenza) * dt
        self.update_size()
        self.update_rect()
        if self.destroy():
            return True
        return False

    def destroy(self):
        if self.rect.top > 750:
            self.kill()
            return True
        return False

    def update_size(self):
        if self.render_mode != "human":
            return
        scale_factor = 0.2 + self.t * (1 - 0.2)

        new_width = int(200 * scale_factor)
        new_height = int(180 * scale_factor)

        self.image = pygame.transform.scale(
            self.original_image,
            (new_width, new_height)
        )

    def update_rect(self):
        lane_x_left = lane_interpolation(lanes[self.colore_nota][0][0], lanes[self.colore_nota][2][0], self.t)
        lane_x_right = lane_interpolation(lanes[self.colore_nota][1][0], lanes[self.colore_nota][3][0], self.t)

        target_y = lane_interpolation(252, 750, self.t)

        center_x = (lane_x_left + lane_x_right) / 2

        if self.render_mode == "human":
            self.rect = self.image.get_rect(center=(int(center_x), int(target_y)))
        else:
            # In caso di esecuzione AI, aggiorniamo semplicemente le coordinate del rect esistente
            self.rect.center = (int(center_x), int(target_y))

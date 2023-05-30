import os
import pygame
import random
import cv2
import csv
from pygame.locals import *

pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([0, 0], pygame.FULLSCREEN)

# Run until the user asks to quit
running = True
circle_size = 12
max_x, max_y = pygame.display.get_surface().get_size()
max_x, max_y = max_x - circle_size, max_y - circle_size # 1524, 852
mid_max_x, mid_max_y = (max_x+12) // 2, (max_y+12) // 2

calibration_coord = [(circle_size, circle_size), (circle_size, mid_max_y), (circle_size, max_y),
                   (mid_max_x, circle_size), (mid_max_x, mid_max_y), (mid_max_x, max_y),
                   (max_x, circle_size), (max_x, mid_max_y), (max_x, max_y)
                   ]

x_choice = [i for i in range(circle_size, max_x+1)]
y_choice = [i for i in range(circle_size, max_y+1)]

webcam = cv2.VideoCapture(1)
ret, frame = webcam.read()
idx = 0 if len(os.listdir('data/_data')) == 0 else len(os.listdir('data/_data')) - 1
coordinate = {}
x, y = 0, 0

random.shuffle(calibration_coord)
while running:
    event = pygame.event.wait()
    if event.type == KEYDOWN and event.key == K_ESCAPE:
        running = False
        pygame.quit()
        break
    elif event.type == KEYDOWN and event.key == K_SPACE:

        if idx < 9:  # get calibrate screen coordinate
            x, y = calibration_coord[idx]
        else:  # get random screen coordinate
            y = random.choice(y_choice)
            x = random.choice(x_choice)

        # fill the background with white
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (0, 0, 255), (x, y), circle_size)

    elif event.type == KEYDOWN and event.key == K_z:
        ret, frame = webcam.read()
        out_path = f"data/_data/{idx}.jpg"
        cv2.imwrite(out_path, frame)

        with open('data/_data/coordinate.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, (x,y,-1)])

        idx += 1

    pygame.event.clear()
    # flip the display
    pygame.display.flip()

webcam.release()
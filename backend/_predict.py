import torch
import os
import json
import numpy as np
import pygame
import random
import csv
import cv2

from torchvision import transforms
from Models import FullModel, SingleModel
from Models import EyesModel
from pygame.locals import *
from _data_preprocess import Detector


class Predictor:
    def __init__(self, model, model_data, config_file=None, gpu=0):
        super().__init__()

        _, ext = os.path.splitext(model_data)
        if ext == ".ckpt":
            self.model = model.load_from_checkpoint(model_data)
        else:
            with open(config_file) as json_file:
                config = json.load(json_file)
            self.model = model(config)
            self.model.load_state_dict(torch.load(model_data))

        self.gpu = gpu
        self.model.double()
        self.model.cuda(self.gpu)
        self.model.eval()

    def predict(self, *img_list, head_angle=None):
        images = []
        print(len(img_list))
        for i, img in enumerate(img_list):
            if not img.dtype == np.uint8:
                img = img.astype(np.uint8)
            img = transforms.ToTensor()(img).unsqueeze(0)
            img = img.double()
            img = img.cuda(self.gpu)
            images.append(img)

        if head_angle is not None:
            angle = torch.tensor(head_angle).double().flatten().cuda(self.gpu)
            images.append(angle)

        with torch.no_grad():
            coords = self.model(*images)
            coords = coords.cpu().numpy()[0]

        return coords[0], coords[1]


if __name__ == "__main__":

    ''' start to generate test data'''
    pygame.init()

    # Set up the drawing window
    screen = pygame.display.set_mode([0, 0], pygame.FULLSCREEN)

    # Run until the user asks to quit
    running = True
    circle_size = 12
    max_x, max_y = pygame.display.get_surface().get_size()
    max_x, max_y = max_x - circle_size, max_y - circle_size  # 1524, 852
    mid_max_x, mid_max_y = (max_x + 12) // 2, (max_y + 12) // 2

    calibration_coord = [(circle_size, circle_size), (circle_size, mid_max_y), (circle_size, max_y),
                         (mid_max_x, circle_size), (mid_max_x, mid_max_y), (mid_max_x, max_y),
                         (max_x, circle_size), (max_x, mid_max_y), (max_x, max_y)
                         ]

    x_choice = [i for i in range(circle_size, max_x + 1)]
    y_choice = [i for i in range(circle_size, max_y + 1)]

    webcam = cv2.VideoCapture(1)
    ret, frame = webcam.read()
    # idx = 0 if len(os.listdir('test_data/_data')) == 0 else len(os.listdir('test_data/_data')) - 1
    idx = 0
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

            y = random.choice(y_choice)
            x = random.choice(x_choice)
            # x, y = calibration_coord[idx]

            # fill the background with white
            screen.fill((255, 255, 255))
            pygame.draw.circle(screen, (0, 255, 0), (x, y), circle_size)

        elif event.type == KEYDOWN and event.key == K_z:
            ret, frame = webcam.read()
            out_path = f"test_data/_data/{idx}.jpg"
            cv2.imwrite(out_path, frame)

            with open('test_data/_data/coordinate.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([idx, (x, y, -1)])

            idx += 1

        pygame.event.clear()
        # flip the display
        pygame.display.flip()

    webcam.release()

    ''' start to process image and get estimated coordination '''
    detector = Detector(
        output_size=64
    )

    path = "test_data/_data"
    data_file_path = "test_data/labeling.csv"

    data_file = open(data_file_path, "a", newline="")
    csv_writer = csv.writer(data_file, delimiter=",")

    predictor = Predictor(
        EyesModel,
        model_data="trained_models/eyetracking_eye_model.pt",
        config_file="trained_models/eyetracking_eye_config.json",
    )

    result = []
    img_list = os.listdir(f"test_data/_data")
    for idx in range(len(img_list) - 1):
        frame = cv2.imread(f"test_data/_data/{idx}.jpg")
        detector.get_frame(frame)
        x, y = predictor.predict(detector.l_eye_img, detector.r_eye_img)
        result.append((x, y))

    for x, y in result:
        print(x, y)

    idx = 0
    pygame.init()
    running = True
    # Set up the drawing window
    screen = pygame.display.set_mode([0, 0], pygame.FULLSCREEN)
    circle_size = 12
    max_x, max_y = pygame.display.get_surface().get_size()
    max_x, max_y = max_x - circle_size, max_y - circle_size  # 1524, 852
    mid_max_x, mid_max_y = (max_x + 12) // 2, (max_y + 12) // 2

    calibration_coord = [(circle_size, circle_size), (circle_size, mid_max_y), (circle_size, max_y),
                         (mid_max_x, circle_size), (mid_max_x, mid_max_y), (mid_max_x, max_y),
                         (max_x, circle_size), (max_x, mid_max_y), (max_x, max_y)
                         ]
    while running:
        event = pygame.event.wait()
        if event.type == KEYDOWN and event.key == K_ESCAPE:
            running = False
            pygame.quit()
            break
        elif event.type == KEYDOWN and event.key == K_SPACE:

            x, y = map(int, result[idx])

            # fill the background with white
            screen.fill((255, 255, 255))
            pygame.draw.circle(screen, (0, 255, 0), (x, y), circle_size)

            idx += 1
            if idx == 8:
                break
        elif event.type == KEYDOWN and event.key == K_c:
            print(idx)

        pygame.event.clear()
        # flip the display
        pygame.display.flip()


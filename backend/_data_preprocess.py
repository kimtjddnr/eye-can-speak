import os
import cv2
import dlib
import numpy as np
import csv

from collections import OrderedDict
from utils import get_config, shape_to_np

# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")


class Detector:
    def __init__(self, output_size, gpu=0):
        dlib.DLIB_USE_CUDA = True
        self.output_size = output_size
        self.face_img = np.zeros((output_size, output_size, 3))
        self.face_align_img = np.zeros((output_size, output_size, 3))
        self.l_eye_img = np.zeros((output_size, output_size, 3))
        self.r_eye_img = np.zeros((output_size, output_size, 3))
        self.head_pos = np.ones((output_size, output_size))
        self.head_angle = 0.0

        dlib.cuda.set_device(gpu)

        self.landmark_idx = OrderedDict([("right_eye", (0, 2)), ("left_eye", (2, 4))])
        self.detector = dlib.cnn_face_detection_model_v1(
            "trained_models/mmod_human_face_detector.dat"
        )
        self.predictor = dlib.shape_predictor(
            "trained_models/shape_predictor_5_face_landmarks.dat"
        )

    def get_frame(self, frame):
        dets = self.detector(frame, 0)

        if len(dets) == 1:
            features = self.predictor(frame, dets[0].rect)
            reshaped = shape_to_np(features)

            l_start, l_end = self.landmark_idx["left_eye"]
            r_start, r_end = self.landmark_idx["right_eye"]
            l_eye_pts = reshaped[l_start:l_end]
            r_eye_pts = reshaped[r_start:r_end]
            l_eye_width = l_eye_pts[1][0] - l_eye_pts[0][0]
            r_eye_width = r_eye_pts[0][0] - r_eye_pts[1][0]

            # Calculate eye centers and head angle
            l_eye_center = l_eye_pts.mean(axis=0).astype("int")
            r_eye_center = r_eye_pts.mean(axis=0).astype("int")
            # vector normalization
            eye_dist = np.linalg.norm(r_eye_center - l_eye_center)
            dY = r_eye_center[1] - l_eye_center[1]
            dX = r_eye_center[0] - l_eye_center[0]
            self.head_angle = np.degrees(np.arctan2(dY, dX))

            # Face extraction and alignment
            desired_l_eye_pos = (0.35, 0.5)
            desired_r_eye_posx = 1.0 - desired_l_eye_pos[0]

            desired_dist = desired_r_eye_posx - desired_l_eye_pos[0]
            desired_dist *= self.output_size
            scale = desired_dist / eye_dist

            eyes_center = (
                (l_eye_center[0] + r_eye_center[0]) / 2,
                (l_eye_center[1] + r_eye_center[1]) / 2,
            )

            t_x = self.output_size * 0.5
            t_y = self.output_size * desired_l_eye_pos[1]

            align_angles = (0, self.head_angle)
            for angle in align_angles:
                M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
                M[0, 2] += t_x - eyes_center[0]
                M[1, 2] += t_y - eyes_center[1]

                aligned = cv2.warpAffine(
                    frame,
                    M,
                    (self.output_size, self.output_size),
                    flags=cv2.INTER_CUBIC,
                )

                if angle == 0:
                    self.face_img = aligned
                else:
                    self.face_align_img = aligned
            # Get eyes (square regions based on eye width)
            try:
                l_eye_img = frame[
                    l_eye_center[1]
                    - int(l_eye_width / 2) : l_eye_center[1]
                    + int(l_eye_width / 2),
                    l_eye_pts[0][0] : l_eye_pts[1][0],
                ]
                self.l_eye_img = cv2.resize(
                    l_eye_img, (self.output_size, self.output_size)
                )

                r_eye_img = frame[
                    r_eye_center[1]
                    - int(r_eye_width / 2) : r_eye_center[1]
                    + int(r_eye_width / 2),
                    r_eye_pts[1][0] : r_eye_pts[0][0],
                ]
                self.r_eye_img = cv2.resize(
                    r_eye_img, (self.output_size, self.output_size)
                )
            except:
                pass

            # Get position of head in the frame
            frame_bw = np.ones((frame.shape[0], frame.shape[1])) * 255
            cv2.rectangle(
                frame_bw,
                (dets[0].rect.left(), dets[0].rect.top()),
                (dets[0].rect.right(), dets[0].rect.bottom()),
                COLOURS["black"],
                -1,
            )
            self.head_pos = cv2.resize(frame_bw, (self.output_size, self.output_size))

    def save_data(self, data_id, coord_x, coord_y):
        for (path, img) in zip(data_dirs, (self.l_eye_img, self.r_eye_img, self.face_img, self.face_align_img, self.head_pos)):
            cv2.imwrite("{}/{}.jpg".format(path, data_id), img)
        csv_writer.writerow([data_id, coord_x, coord_y, self.head_angle])


if __name__ == "__main__":
    data_dirs = (
        "./data/l_eye",
        "./data/r_eye",
        "./data/face",
        "./data/face_aligned",
        "./data/head_pos",
    )
    data_file_path = "./data/labeling.csv"
    data_file_exists = os.path.isfile(data_file_path)

    data_file = open(data_file_path, "a", newline="")
    csv_writer = csv.writer(data_file, delimiter=",")

    if not data_file_exists:
        csv_writer.writerow(["id", "x", "y", "head_angle"])

    with open(f"./data/_data/coordinate.csv", "r") as f_in:
        reader = csv.reader(f_in)
        with open("./data/labeling.csv", "a") as f_out:
            writer = csv.writer(f_out)
            coordinate = {int(row[0]): tuple(int(val.strip()) for val in row[1].strip('()').split(',')) for row in
                          reader}

    detector = Detector(
        output_size=64
    )

    file_list = os.listdir(f"./data/_data")
    for idx in range(len(file_list)-1):
        coord_x, coord_y, _ = coordinate[idx]
        img = cv2.imread(f"./data/_data/{idx}.jpg")

        print(f"{idx} data processing...")
        detector.get_frame(img)
        print(f"{idx} data saving...")
        detector.save_data(idx, coord_x, coord_y)

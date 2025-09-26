"""
Air handwriting + character classifier
- OpenCV for camera
- MediaPipe for hand/finger tracking
- TensorFlow/Keras for a simple CNN

Controls while running (camera window focused):
 - 0-9 : save current canvas as label '0'..'9' into dataset/ (creates folders)
 - c   : clear canvas
 - t   : train CNN on images in dataset/ (blocks while training)
 - p   : predict current canvas using saved model 'model.h5'
 - q   : quit
 - s   : save current canvas as PNG (unnamed) into saved/

Notes:
 - Install requirements: pip install opencv-python mediapipe tensorflow numpy
 - This script is intentionally simple and educational.
 - For letters beyond digits, change the saving keys and folders.

"""

import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ---- Parameters ----
CAM_W, CAM_H = 640, 480
CANVAS_W, CANVAS_H = 280, 280  # working canvas (keep aspect), we'll downscale to 28x28 for the model
DRAW_COLOR = (255, 255, 255)  # white on black canvas
LINE_THICKNESS = 8
DATASET_DIR = 'dataset'
MODEL_PATH = 'model.h5'
SAVED_DIR = 'saved'

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(SAVED_DIR, exist_ok=True)

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ---- Helper functions ----

def finger_up(landmarks, finger_tip_idx, finger_dip_idx):
    """Return True if given finger appears up (tip is above dip in y coordinate in image coords)."""
    # Note: image coords: y increases downward. So tip.y < dip.y means finger extended upwards.
    tip = landmarks[finger_tip_idx]
    dip = landmarks[finger_dip_idx]
    return tip.y < dip.y


def build_simple_cnn(input_shape=(28,28,1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_dataset(img_size=(28,28)):
    X, y = [], []
    labels = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    for lab in labels:
        lab_dir = os.path.join(DATASET_DIR, lab)
        for fn in os.listdir(lab_dir):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(lab_dir, fn)
            img = load_img(path, color_mode='grayscale', target_size=img_size)
            arr = img_to_array(img) / 255.0
            X.append(arr)
            y.append(label_to_idx[lab])
    if len(X) == 0:
        return None, None, None
    X = np.array(X, dtype='float32')
    y = to_categorical(np.array(y), num_classes=len(labels))
    # reshape for channel
    if X.ndim == 3:
        X = X.reshape((*X.shape, 1))
    return X, y, labels


def train_and_save_model():
    print('\n[TRAIN] Loading dataset...')
    X, y, labels = load_dataset()
    if X is None:
        print('[TRAIN] No data found in dataset/. Save some examples first by pressing a digit key (0-9).')
        return
    num_classes = y.shape[1]
    model = build_simple_cnn(input_shape=(28,28,1), num_classes=num_classes)
    print(f'[TRAIN] Training on {X.shape[0]} samples, classes: {labels}')
    model.fit(X, y, epochs=12, batch_size=16, validation_split=0.15)
    # save model and labels
    model.save(MODEL_PATH)
    with open('labels.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))
    print(f'[TRAIN] Model saved to {MODEL_PATH}, labels saved to labels.txt')


def predict_canvas(canvas):
    if not os.path.exists(MODEL_PATH) or not os.path.exists('labels.txt'):
        print('[PREDICT] Trained model or labels.txt not found. Train first with t key.')
        return
    labels = [l.strip() for l in open('labels.txt', encoding='utf-8').read().splitlines() if l.strip()]
    model = load_model(MODEL_PATH)
    img = cv2.resize(canvas, (28,28))
    img = cv2.bitwise_not(img)  # we drew white on black; invert if needed so digits are dark on light
    arr = img_to_array(img).astype('float32') / 255.0
    arr = arr.reshape((1,28,28,1))
    proba = model.predict(arr)[0]
    idx = np.argmax(proba)
    print(f"[PREDICT] -> {labels[idx]} (p={proba[idx]:.2f})")

# ---- Main live loop ----

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # Black canvas for drawing (CANVAS_W x CANVAS_H)
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    prev_x, prev_y = None, None

    smoothing = []  # keep last few points to smooth

    print('Starting camera. Controls: 0-9 save label, c clear, s save png, t train, p predict, q quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h, w, _ = frame.shape

        drawing = frame.copy()

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(drawing, lm, mp_hands.HAND_CONNECTIONS)
            landmarks = lm.landmark

            # check index finger up
            index_up = finger_up(landmarks, 8, 6)  # tip idx 8, pip/dip idx 6
            # optionally check middle finger to decide mode, but we'll draw when index is up

            if index_up:
                x = int(landmarks[8].x * CANVAS_W)
                y = int(landmarks[8].y * CANVAS_H)

                smoothing.append((x,y))
                if len(smoothing) > 5:
                    smoothing.pop(0)
                sx = int(sum([p[0] for p in smoothing]) / len(smoothing))
                sy = int(sum([p[1] for p in smoothing]) / len(smoothing))

                if prev_x is None:
                    prev_x, prev_y = sx, sy
                cv2.line(canvas, (prev_x, prev_y), (sx, sy), DRAW_COLOR[0], LINE_THICKNESS)
                prev_x, prev_y = sx, sy
            else:
                prev_x, prev_y = None, None
                smoothing = []

        else:
            prev_x, prev_y = None, None
            smoothing = []

        # show windows: small canvas preview on the frame
        canvas_resized = cv2.resize(canvas, (200,200))
        # make BGR version for overlay
        canvas_col = cv2.cvtColor(canvas_resized, cv2.COLOR_GRAY2BGR)
        # place on top-left corner
        drawing[10:10+200, 10:10+200] = canvas_col

        cv2.imshow('Air Write - press q to quit', drawing)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.fill(0)
            print('[ACTION] Canvas cleared')
        elif key == ord('s'):
            fn = os.path.join(SAVED_DIR, f'shot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            cv2.imwrite(fn, canvas)
            print(f'[ACTION] Canvas saved to {fn}')
        elif key == ord('t'):
            train_and_save_model()
        elif key == ord('p'):
            predict_canvas(canvas)
        elif ord('0') <= key <= ord('9'):
            label = chr(key)
            labdir = os.path.join(DATASET_DIR, label)
            os.makedirs(labdir, exist_ok=True)
            fn = os.path.join(labdir, f'{label}_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}.png')
            # save a 28x28 version for the dataset
            img = cv2.resize(canvas, (28,28))
            cv2.imwrite(fn, img)
            print(f'[ACTION] Saved to {fn}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

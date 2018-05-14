import tensorflow as tf
import cv2
import numpy as np

def parse_text(line):
    out = tf.string_split([line], delimiter=" ").values
    filename, label = out[0], out[1]
    label = tf.string_to_number(label, tf.int32)
    return filename, label

# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, label, max_frames=0, resize=(224, 224)):
    path = "/data/UCF-101/"+path.decode()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames, dtype=np.float32) / 255.0, label

def train_dataset():
    files = ["/data/ucfTrainTestlist/train_map.txt"]
    #ds = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=4))
    ds = tf.data.TextLineDataset(files)
    ds = ds.map(parse_text)
    ds = ds.map(lambda filename, label: tuple(tf.py_func(
        load_video, [filename, label], [tf.float32, label.dtype])))
    return ds

def test_dataset():
    files = ["/data/ucfTrainTestlist/test_map.txt"]
    #ds = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=4))
    ds = tf.data.TextLineDataset(files)
    ds = ds.map(parse_text)
    ds = ds.map(lambda filename, label: tuple(tf.py_func(
        load_video, [filename, label], [tf.float32, label.dtype])))
    return ds

def get_ds():
    files = tf.data.Dataset.list_files("/data/ucfTrainTestlist/train*")
    ds = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=4))
    ds = ds.map(parse_text)
    ds = ds.map(
    lambda filename, label: tuple(tf.py_func(
        load_video, [filename, label], [tf.float32, label.dtype])))
    return ds
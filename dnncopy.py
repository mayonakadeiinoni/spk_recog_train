#!/usr/bin/env python

import joblib
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import shutil
import torch
from silero_vad import read_audio
from scipy.io import wavfile
import scipy.signal
import scipy.signal as signal
import librosa.display
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xvector_jtubespeech import XVector
from torchaudio.compliance import kaldi
import torch
import random
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator, collect_chunks
import json
import scipy.io.wavfile as wavfile
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
sys.stdout.reconfigure(line_buffering=True)

# Xvector 周り

# グローバルモデルの定義
global xvec
xvec = XVector(
    "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/model/xvector.pth"
)


def One_xvector(wav_file, wav_exits=False):
    # wav_file wavファイルのパス
    # 出力 ベクトル[512]次元
    global xvec
    if wav_exits:
        wav = wav_file
    else:
        wav, _ = librosa.load(wav_file, sr=16000)
    wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    mfcc = kaldi.mfcc(
        wav,
        num_ceps=24,
        num_mel_bins=24,
    )
    mfcc = mfcc.unsqueeze(0)
    xvector = xvec.vectorize(mfcc)
    xvector = xvector.to("cpu").detach().numpy().copy()

    return xvector[0]  # [ 512]


# VAD
global vad
vad = load_silero_vad()
# 無音削減


def remove_silence_with_vad(wav, model=vad, sampling_rate=16000, threshold=0.5):
    # Load audio

    # Initialize VAD iterator
    vad_iterator = VADIterator(
        vad, sampling_rate=sampling_rate, threshold=threshold)

    window_size_samples = 512 if sampling_rate == 16000 else 256
    speech_segments = []
    start_list = []
    end_list = []
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i + window_size_samples]
        if len(chunk) < window_size_samples:
            break

        speech_dict = vad_iterator(chunk, return_seconds=False)

        if speech_dict:
           # print(speech_dict)
            if 'start' in speech_dict:
               #     print(55)
                start_list.append(speech_dict['start'])
            elif 'end' in speech_dict:
                end_list.append(speech_dict['end'])
            else:
                pass

    vad_iterator.reset_states()

    # Concatenate speech segments into a single waveform
    wav_reconlist = []
    if start_list != []:

        for start, end in zip(start_list, end_list):
            wav_reconlist.append(wav[start:end])
    wav_recon = np.concatenate(wav_reconlist) if wav_reconlist != [] else []
   # print(wav_recon)
    """
    if wav_recon.size > 0:
        output_filename="uhi.wav"
        wavfile.write(output_filename, sampling_rate, wav_recon)
        print(f"Processed audio saved as {output_filename}")
    else:
        print("No speech detected, no file saved.")
    exit(0)
    """
    return wav_recon, sampling_rate

# DNNモデル定義


class SpeakerRecognitionNN(nn.Module):
    def __init__(self, input_dim, num_speakers):
        super(SpeakerRecognitionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # 1層目
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)  # 2層目
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)  # 3層目
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)

        self.fc_out = nn.Linear(128, num_speakers)  # 出力層
        self.softmax = nn.Softmax(dim=1)  # 確率を出力

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)

        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)

        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc_out(x)
        return x  # 話者ごとの確率を出力


# Model初期化
# ハイパーパラメータ設定
# ======== 設定 ========
INPUT_DIM = 512  # x-vector の次元数
NUM_SPEAKERS = 12  # 話者の数（base_dirs の数と対応）

LEARNING_RATE = 0.001
SAMPLING_RATE = 16000
BATCH_SIZE = 32
# モデルの初期化
model = SpeakerRecognitionNN(INPUT_DIM, NUM_SPEAKERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


class Trainer:
    def __init__(self, base_root, speaker_names):
        self.base_root = base_root
        self.speaker_names = speaker_names
        self.scaler = StandardScaler()
        self.label_map = {name: i for i, name in enumerate(speaker_names)}
        self.a = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0]
        print(self.a)

        split_data = self.collect_split_files()

        """
          [{wav,label_num}]
        """
        self.test_files = split_data['test']
        self.train_files = split_data['train']
        self.dev_files = split_data['dev']
        """
          [
              {length:
              {vector,label_num}
              }
          ]
        """

        # 初期学習データの初期化
        self.init_X_train, self.init_y_train = self.generate_batch_data(
            # self.train_files, num_segments=2, select_num=2, Data_num=2
            self.train_files, num_segments=1000, select_num=500, Data_num=7000

        )
        print(f"self.X_train:{len(self.init_X_train)}")

        self.scaler.fit(self.init_X_train)

        self.dev_lists = self.eval_set_generate(self.dev_files)
        self.test_lists = self.eval_set_generate(self.test_files)

    # 話者ごとの train/dev/test ファイルを集める

    def collect_split_files(self):
        label_map = self.label_map
        split_data = {
            "train": [],
            "dev": [],
            "test": []
        }

        for name in self.speaker_names:
            for split in ["train", "dev", "test"]:
                split_dir = os.path.join(self.base_root, name, split)
            #  print(split_dir)
                if not os.path.exists(split_dir):
                    continue
                for file in os.listdir(split_dir):
                    #  print(file)
                    if file.endswith(".wav"):
                        full_path = os.path.join(split_dir, file)
                        split_data[split].append((full_path, label_map[name]))

        print(
            f"Train: {len(split_data['train'])}, Dev: {len(split_data['dev'])}, Test: {len(split_data['test'])}")
        return split_data

    def eval_set_generate(self, testfiles):
        test_list = {}
        a = self.a
        a.reverse()
        for test_segment_len in a:
            X_test, y_test = [], []
            for file_path, label in self.test_files:

                wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
                wav, _ = remove_silence_with_vad(wav)

                step = int(test_segment_len * SAMPLING_RATE)
                for i in range(0, len(wav) - step, step):
                    segment = wav[i:i + step]
                    features = One_xvector(segment, wav_exits=True)
                    X_test.append(features)
                    y_test.append(label)

            test_list[test_segment_len] = {
                "X_test": self.scaler.transform(X_test), "y_test": np.array(y_test)}

        return test_list

    def generate_batch_data(self, train_files, num_segments=100, select_num=500, Data_num=5000):
        X_train, y_train = [], []
        speaker_segments = {}
        # wavファイルのランダム切り出し
        for full_path, speaker_label in train_files:
         #   print(full_path)
            wav = read_audio(full_path, sampling_rate=SAMPLING_RATE)
        #    print(len(wav))
            wav, _ = remove_silence_with_vad(wav)
            # データ拡張
         #   print(len(wav))
            if len(wav) == 0 or len(wav) <= 3000:
                continue

            for _ in range(num_segments):
                start = random.randint(0, max(0, len(wav) - 3000))
                length = random.randint(3000, min(80000, len(wav) - start))
                segment = wav[start:start + length]

                label = speaker_label
                if label not in speaker_segments:
                    speaker_segments[label] = []
         #       print(f"NOJ")
                speaker_segments[label].append(segment)
        # 　ランダム切り出しから結合
        # print(f"speaker_segments:{len(speaker_segments)}")
        a = 0
        while (a <= Data_num+5):
            for speaker, segments in speaker_segments.items():
             #       print(f"segments:{len(segments)}")
                num_selected = select_num  # int(BATCH_SIZE / NUM_SPEAKERS)
                selected_segments = random.sample(segments, num_selected)
                new_segment = np.concatenate(selected_segments, axis=0)

                step = int(random.uniform(0.180, 1.5) * SAMPLING_RATE)
                for i in range(0, len(new_segment) - step, step):
                    segment = new_segment[i:i + step]
                    features = One_xvector(segment, wav_exits=True)
                    X_train.append(features)
                    y_train.append(speaker)
            a = len(X_train)
          #  print(a)

        selected_indices = random.sample(range(len(X_train)), Data_num)
        X_train = [X_train[i] for i in selected_indices]
        y_train = [y_train[i] for i in selected_indices]
        # print(f"学習データ:{len(X_train)}")
        return np.array(X_train), np.array(y_train)

    def train_model(self, X_train, y_train, model, optimizer, criterion, epochs=1):
        model.train()

        X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(X_train_tensor)

        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate_model(self, X_test, y_test, model):
        model.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)

        accuracy = (predicted == y_test_tensor).float().mean().item()
        return accuracy, predicted

    def save_model_and_script(self, model, directory):
        save_dir = directory
        os.makedirs(save_dir, exist_ok=True)

        # モデルの保存
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"モデルを保存しました: {model_path}")

        # 実行したPythonスクリプトも保存
        try:
            script_path = os.path.join(
                save_dir, "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/名工大/オンライン学習/train.py")
            shutil.copy(__file__, script_path)
            print(f"実行スクリプトを保存しました: {script_path}")
        except NameError:
            print("警告: __file__が使用できません。Jupyter Notebookで実行中の可能性があります。")


# Data準備
# 話者ディレクトリのベース
base_root = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ"

# 話者名リスト
speaker_names = ["中川", "今村", "佐藤", "山下", "川崎", "柏木",
                 "浅野", "渡邊", "神谷", "苅谷", "薫田", "高木"]  # 人名リスト

trainer = Trainer(base_root=base_root, speaker_names=speaker_names)

# ======== バッチ学習ループ ========

# TensorBoard ログの保存先
save_dir = f"/work/asano.yuki/model/demo/話者認識/2025_04_16/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir = f"{save_dir}/dnn_speaker_recognition"
os.makedirs(log_dir, exist_ok=True)
# 'scaler.pkl' は好きなファイル名でOK
joblib.dump(trainer.scaler, os.path.join(log_dir, 'scaler.pkl'))
# TensorBoard の Writer を作成
writer = SummaryWriter(log_dir)

count = 0
global_step = 0
while True:
    if count == 0:
        # 新しいバッチデータを生成
        X_train, y_train = trainer.init_X_train, trainer.init_y_train
        X_train = trainer.scaler.transform(X_train)
        count = 1
    else:
        # 新しいバッチデータを生成
        X_train, y_train = trainer.generate_batch_data(
            # trainer.train_files, num_segments=2, select_num=2, Data_num=2

            trainer.train_files, num_segments=500, select_num=200, Data_num=3000

        )
        X_train = trainer.scaler.transform(X_train)
    # === ニューラルネットワーク学習 ===
    BATCH_SIZE = len(X_train)
    for i in range(0, len(X_train), BATCH_SIZE):
        batch_X = X_train[i:i+BATCH_SIZE]
        batch_y = y_train[i:i+BATCH_SIZE]
        loss = trainer.train_model(
            X_train, y_train, model, optimizer, criterion)
        # ★ LossをTensorBoardに記録
        writer.add_scalar("Train/Loss", loss, global_step)
        global_step += 1
       # print(f"Loss: {loss:.4f}")
    # === テストデータでの評価 ===
    all_above_threshold = True

    for test_segment_len in trainer.a:
        X_dev, y_dev = trainer.scaler.transform(
            trainer.dev_lists[test_segment_len]["X_test"]), trainer.dev_lists[test_segment_len]["y_test"]
        accuracy, y_pred = trainer.evaluate_model(X_dev, y_dev, model)
        print(
            f"Test Segment {test_segment_len}s Accuracy: {accuracy * 100:.2f}%")
        # TensorBoard に正解率を記録
        # ★ AccuracyをTensorBoardに記録（セグメント長ごと）
        writer.add_scalar(
            f"Dev Accuracy/{test_segment_len}s", accuracy, global_step)

        if accuracy >= 0.95 and test_segment_len == 0.5:
            print("0.5のテストデータで98%以上の精度を達成！一時保存します。")
            trainer.save_model_and_script(model, directory=os.path.join(
                save_dir, "dnn.model"))

            X_test, y_test = trainer.scaler.transform(
                trainer.test_lists[test_segment_len]["X_test"]), trainer.test_lists[test_segment_len]["y_test"]
            accuracy, y_pred = trainer.evaluate_model(X_test, y_test, model)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")

            print(confusion_matrix(y_test, y_pred))
        if accuracy < 0.98:
            all_above_threshold = False

    if all_above_threshold:
        print("すべてのテストデータで98%以上の精度を達成！終了します。")
        trainer.save_model_and_script(model, directory=os.path.join(
            save_dir, "dnn.model"))
        break

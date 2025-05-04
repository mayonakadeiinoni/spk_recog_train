#!/usr/bin/env python
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import shutil
import torch
from silero_vad import read_audio
from scipy.io import wavfile
#!/usr/bin/env python
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
sys.stdout.reconfigure(line_buffering=True)

random.seed(42)
# 特徴量抽出関数

# グローバルモデルの定義
global xvec
xvec = XVector(
    "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/model/xvector.pth"
)
global vad
vad = load_silero_vad()


# 音声ファイルからXベクトルを取得する関数（既存の関数）


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
    wav_recon = np.concatenate(wav_reconlist) if wav_reconlist != [] else False
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


# クラス定義


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


# ======== 設定 ========
INPUT_DIM = 512  # x-vector の次元数
NUM_SPEAKERS = 3  # 話者の数（base_dirs の数と対応）
SAMPLING_RATE = 16000
BATCH_SIZE = 10000

LEARNING_RATE = 0.3  # XGBoost の学習率
NUM_ROUNDS = 10  # 初回のブースティング回数
BATCH_UPDATE_ROUNDS = 1  # バッチごとの更新回数

# 話者のディレクトリ
base_dirs = [
    "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/名工大/高木ToWav",
    "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/名工大/山下ToWav",
    "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/名工大/柏木ToWav"
]

# ディレクトリごとのラベルマッピング
label_map = {j: i for i, j in enumerate(base_dirs)}

# 共通する wav ファイル名を取得
file_sets = [set(os.listdir(d)) for d in base_dirs]
common_files = sorted(set.intersection(*file_sets))
test_size = int(len(common_files) * 0.35)
test_files = random.sample(common_files, test_size)
train_files = list(set(common_files) - set(test_files))

print(f"Train: {len(train_files)}, Test: {len(test_files)}")


# ハイパーパラメータ設定
INPUT_DIM = 512  # x-vectorの次元数（One_xvectorの出力サイズ）
NUM_SPEAKERS = len(label_map)  # 話者の総数
LEARNING_RATE = 0.001
SAMPLING_RATE = 16000
BATCH_SIZE = int(10000)

# モデルの初期化
model = SpeakerRecognitionNN(INPUT_DIM, NUM_SPEAKERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_model(X_train, y_train, model, optimizer, criterion, epochs=1):
    model.train()

    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model(X_test, y_test, model):
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test_tensor).float().mean().item()
    return accuracy


# ======== モデルの初期化 ========
scaler = StandardScaler()
params = {
    'objective': 'multi:softmax',
    'num_class': NUM_SPEAKERS,
    'max_depth': 6,
    'eta': LEARNING_RATE,
    'eval_metric': 'mlogloss',
    'tree_method': 'hist'
}

# ======== XGBoost 用のデータ生成関数 ========


def generate_batch_data(speaker_segments, select_num=5000, Data_num=50000):
    """XGBoost 用のバッチデータを生成"""
    X_train, y_train = [], []
    # 欲しいデータ数
    a = 0
    while (a <= Data_num+5):
        for speaker, segments in speaker_segments.items():
            num_selected = select_num  # int(BATCH_SIZE / NUM_SPEAKERS)
            selected_segments = random.sample(segments, num_selected)
            new_segment = np.concatenate(selected_segments, axis=0)

            step = int(random.uniform(0.180, 1.2) * SAMPLING_RATE)
            for i in range(0, len(new_segment) - step, step):
                segment = new_segment[i:i + step]
                features = One_xvector(segment, wav_exits=True)
                X_train.append(features)
                y_train.append(speaker)
        a = len(X_train)

    selected_indices = random.sample(range(len(X_train)), Data_num)
    X_train = [X_train[i] for i in selected_indices]
    y_train = [y_train[i] for i in selected_indices]
    print(f"学習データ:{len(X_train)}")
    return np.array(X_train), np.array(y_train)

# モデル保存用ディレクトリ作成関数


def save_model_and_script(model, directory):
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


# tensoboard の準備


# TensorBoard ログの保存先
save_dir = f"/work/asano.yuki/model/demo/話者認識/2025_04_16/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir = f"{save_dir}/dnn_speaker_recognition"
os.makedirs(log_dir, exist_ok=True)


# ======== 学習データの準備 ========
speaker_segments = {}
for directory in base_dirs:
    for file_name in train_files:
        file_path = os.path.join(directory, file_name)
        wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
        wav, _ = remove_silence_with_vad(wav)

        num_segments = 100
        for _ in range(num_segments):
            start = random.randint(0, max(0, len(wav) - 3000))
            length = random.randint(3000, min(80000, len(wav) - start))
            segment = wav[start:start + length]

            label = label_map[directory]
            if label not in speaker_segments:
                speaker_segments[label] = []
            speaker_segments[label].append(segment)


# TensorBoard の Writer を作成
writer = SummaryWriter(log_dir)

# 初回データの取得
X_train, y_train = generate_batch_data(
    # speaker_segments, select_num=10, Data_num=100)
    speaker_segments, select_num=100, Data_num=500)
scaler.fit(X_train)


# ======== テストデータの準備 ========
test_list = {}
a = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0]
a.reverse()
for test_segment_len in a:
    X_test, y_test = [], []
    for file_name in test_files:
        for directory in base_dirs:
            file_path = os.path.join(directory, file_name)
            wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
            wav, _ = remove_silence_with_vad(wav)

            step = int(test_segment_len * SAMPLING_RATE)
            for i in range(0, len(wav) - step, step):
                segment = wav[i:i + step]
                features = One_xvector(segment, wav_exits=True)
                X_test.append(features)
                y_test.append(label_map[directory])

    test_list[test_segment_len] = {
        "X_test": scaler.transform(X_test), "y_test": np.array(y_test)}

# ======== バッチ学習ループ ========
while True:
    # 新しいバッチデータを生成
    X_train, y_train = generate_batch_data(
        speaker_segments, select_num=100, Data_num=500)
    X_train = scaler.transform(X_train)
    # === ニューラルネットワーク学習 ===
    loss = train_model(X_train, y_train, model, optimizer, criterion)
    print(f"Loss: {loss:.4f}")
    # === テストデータでの評価 ===
    all_above_threshold = True

    for test_segment_len in a:
        X_test, y_test = scaler.transform(
            test_list[test_segment_len]["X_test"]), test_list[test_segment_len]["y_test"]
        accuracy = evaluate_model(X_test, y_test, model)
        print(
            f"Test Segment {test_segment_len}s Accuracy: {accuracy * 100:.2f}%")
        # TensorBoard に正解率を記録
        writer.add_scalar(
            f"Test Accuracy/{test_segment_len}s", accuracy, global_step=len(writer.all_writers))

        if accuracy < 0.98:
            all_above_threshold = False
            break
        if accuracy >= 0.98 and test_segment_len == 0.5:
            print("0.5のテストデータで98%以上の精度を達成！一時保存します。")
            save_model_and_script(model, directory=os.path.join(
                save_dir, "xgb_speaker_recognition.model"))

    if all_above_threshold:
        print("すべてのテストデータで98%以上の精度を達成！終了します。")
        save_model_and_script(model, directory=os.path.join(
            save_dir, "xgb_speaker_recognition.model"))
        break

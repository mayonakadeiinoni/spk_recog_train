#!/usr/bin/env python
import torch.nn.functional as F
import math
import datetime
from torch.utils.data.sampler import WeightedRandomSampler
import joblib
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import librosa.display
import scipy.signal as signal
import scipy.signal
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
from datetime import datetime
import soundfile as sf
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
import sys
sys.stdout.reconfigure(line_buffering=True)

#!/usr/bin/env python


def augment_audio_for_speaker(wav, sr):
    """話者認識に適したデータ拡張（軽微な変化に留める）"""
    augmented_list = [wav]

    # 1. ノイズ追加（低SNR）
    noise_amp = 0.005 * np.random.uniform(0.9, 1.1)
    noise = noise_amp * np.random.randn(len(wav))
    wav_noise = wav + noise
    augmented_list.append(wav_noise)

    # 2. ピッチシフト（±1 semitone）
    for steps in [-0.5, 0.5]:
        try:
            shifted = librosa.effects.pitch_shift(wav, sr=sr, n_steps=steps)
            augmented_list.append(shifted)
        except Exception:
            pass

    # 3. 音量変化（±2dB）
    for gain_db in [-2, -1, 1, 2]:
        factor = 10 ** (gain_db / 20)
        augmented_list.append(wav * factor)

    # 5. ローパス・ハイパス（どちらかランダム）
    from random import choice
    filt_type = choice(["low", "high"])
    cutoff = 5000 if filt_type == "low" else 500
    sos = scipy.signal.butter(10, cutoff, btype=filt_type, fs=sr, output='sos')
    filtered = scipy.signal.sosfilt(sos, wav)
    augmented_list.append(filtered)

    # 6. 軽微なタイムストレッチ（自然に聞こえる範囲）
    for rate in [0.9, 0.95, 1.05, 1.1, 1.2]:

        stretched = librosa.effects.time_stretch(y=wav, rate=rate)
        augmented_list.append(stretched)

    return augmented_list


# from transformers.pipelines.audio_utils import VADIterator
SAMPLING_RATE = 16000
global vad
vad = load_silero_vad()
random.seed(42)


# Data


def collect_split_files(label_map, speaker_names, base_root):
    label_map = label_map
    split_data = {
        "train": [],
        "dev": [],
        "test": []
    }

    for split in ["train", "dev", "test"]:
        a = {}
        for name in speaker_names:
            split_dir = os.path.join(base_root, name, split)
        #  print(split_dir)
            if not os.path.exists(split_dir):
                continue

            for file in os.listdir(split_dir):
                #  print(file)
                if file.endswith(".wav"):
                    full_path = os.path.join(split_dir, file)
                    if label_map[name] not in a:
                        a[label_map[name]] = []
                    a[label_map[name]].append(full_path)

            split_data[split] = a
           # print(a)
           # print(type(split_data[split]))

    print(
        f"Train: {len(split_data['train'])}, Dev: {len(split_data['dev'])}, Test: {len(split_data['test'])}")
    return split_data


##
# 話者ディレクトリのベース
base_root = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ/vad抽出"

# 話者名リスト
speaker_names = ["中川", "今村", "佐藤", "山下", "川崎", "柏木",
                 "浅野", "渡邊", "神谷", "苅谷", "薫田", "高木"]  # 人名リスト
a = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0]
label_map = {j: i for i, j in enumerate(speaker_names)}
split_data = collect_split_files(label_map, speaker_names, base_root)

"""
      [{wav,label_num}]
"""
test_files = split_data['test']
train_files = split_data['train']
dev_files = split_data['dev']


# 特徴量抽出関数

# グローバルモデルの定義
global model
model = XVector(
    "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/model/xvector.pth"
)

# 音声ファイルからXベクトルを取得する関数（既存の関数）


def One_xvector(wav_file, wav_exits=False):
    # wav_file wavファイルのパス
    # 出力 ベクトル[512]次元
    global model
    if wav_exits:
        wav = wav_file
    else:
        wav, _ = librosa.load(wav_file, sr=16000)
 #   wav = normalize_rms(wav)
    wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    mfcc = kaldi.mfcc(
        wav,
        num_ceps=24,
        num_mel_bins=24,
    )
    mfcc = mfcc.unsqueeze(0)
    xvector = model.vectorize(mfcc)
    xvector = xvector.to("cpu").detach().numpy().copy()

    return xvector[0]  # [ 512]


# データセット
X_train, y_train = [], []

target_wav_count = 2  # 10000  # 各話者のターゲットWAV数
print(f"1話者当たりのデータ拡張の割合:{target_wav_count}")
train_segment_len = 0.2
min_length = max((train_segment_len/2) * SAMPLING_RATE,
                 (0.2 * SAMPLING_RATE))
print(f"min_length:{min_length}")
# start_length= int(0.1 * SAMPLING_RATE)
max_length = int(5.0 * SAMPLING_RATE)
print(f"max_length:{max_length}")
print(f"train字のsegmentのlength:{train_segment_len}")
test_segment_len = 0.2
print(f"test字のsegmenのlength:{test_segment_len}")
segment_num_area = [2, 3]  # [3000, 8000]
print(f"各WAVから抽出するセグメント数:{segment_num_area}")
segment_concate_area = [2, 3]  # [100, 500]

X_train, y_train = [], []

save_dir = os.path.join("/work/asano.yuki/model/demo/話者認識/2025_04_16/wav_save",
                        datetime.now().strftime('%Y%m%d_%H%M%S'))

# すべてのWAVファイルを対象にランダムなセグメントを取得
for label in train_files.keys():
    speaker_segments = []  # 話者ごとのセグメント
    for full_path in train_files[label]:
        file_path = os.path.join(full_path)
        wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
        wav = wav.numpy()

        # wav, _ = remove_silence_with_vad(wav)
        if len(wav) == 0 or len(wav) <= 1600*3:
            continue

        # ランダムなセグメントを抽出
        num_segments = random.randint(
            segment_num_area[0], segment_num_area[1])  # 各WAVから抽出するセグメント数
        for _ in range(num_segments):
            start = random.randint(0, max(0, len(wav) - min_length))
            length = random.randint(
                min_length, min(max_length, len(wav) - start))
            segment = wav[start:start + length]

            # --- データ拡張を適用 ---
            augmented_segments = [segment]
           # augmented_segments = augment_audio_for_speaker(
            #    wav=segment, sr=SAMPLING_RATE)

            # 拡張データを追加
            speaker_segments.extend(augmented_segments)

    # 1000個の新しいWAVを作成
    new_speaker_segments = []
    saved_wav_count = 0
    save_sample_count = (5)  # min(10%,20個)
    save_path = os.path.join(
        save_dir, speaker_names[label])
    os.makedirs(save_path, exist_ok=True)

    while len(new_speaker_segments) < target_wav_count:
        num_selected = random.randint(
            segment_concate_area[0], segment_concate_area[1])  # ランダムなセグメント数
        selected_segments = random.sample(speaker_segments, num_selected)
        new_segment = np.concatenate(selected_segments, axis=0)
        new_speaker_segments.append(new_segment)
        # 長さを調整
        if len(new_segment) > max_length:
            new_segment = new_segment[:max_length]
        step = int(train_segment_len * SAMPLING_RATE)
        for i in range(0, len(new_segment) - step, step):
            segment = new_segment[i:i + step]
        #   segment = np.pad(segment, (0, pad), mode='constant')
            features = (One_xvector(segment, wav_exits=True))
            X_train.append(features)
            y_train.append(label)

        step = int(0.35 * SAMPLING_RATE)
        for i in range(0, len(new_segment) - step, step):
            segment = new_segment[i:i + step]
        #   segment = np.pad(segment, (0, pad), mode='constant')
            features = (One_xvector(segment, wav_exits=True))
            X_train.append(features)
            y_train.append(label)
        step = int(0.55 * SAMPLING_RATE)
        for i in range(0, len(new_segment) - step, step):
            segment = new_segment[i:i + step]
        #   segment = np.pad(segment, (0, pad), mode='constant')
            features = (One_xvector(segment, wav_exits=True))
            X_train.append(features)
            y_train.append(label)
        step = int(0.75 * SAMPLING_RATE)
        for i in range(0, len(new_segment) - step, step):
            segment = new_segment[i:i + step]
        #   segment = np.pad(segment, (0, pad), mode='constant')
            features = (One_xvector(segment, wav_exits=True))
            X_train.append(features)
            y_train.append(label)

        # 一部のWAVを保存
        if saved_wav_count < save_sample_count:
            save_file = os.path.join(
                save_path, f"sample_{saved_wav_count}.wav")
            #  write_audio(save_file, new_segment, SAMPLING_RATE)
            sf.write(save_file,  new_segment,  SAMPLING_RATE)
            saved_wav_count += 1
        # データセットに追加
     #   X_train.extend(new_speaker_segments[:target_wav_count])
     #   y_train.extend([label_map[directory]] * target_wav_count)
# print(y_train)
# exit()
X_test, y_test = [], []

for label in dev_files.keys():
    saved_wav_count = 0
    for full_path in dev_files[label]:

        file_path = os.path.join(full_path)

        sr = SAMPLING_RATE
        wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
        wav = wav.numpy()

     #   wav, _ = remove_silence_with_vad(wav)
        re = os.path.join(
            save_dir, speaker_names[label], "dev")
        save_file = os.path.join(
            re, f"{os.path.basename(full_path)}.wav")
        # print(save_file)
        os.makedirs(os.path.join(
            re), exist_ok=True)
        #  write_audio(save_file, new_segment, SAMPLING_RATE)
        sf.write(save_file,  wav,  SAMPLING_RATE)
        saved_wav_count += 1
        if len(wav) == 0 or len(wav) <= 1600*3:
            continue
        step = int(test_segment_len * sr)
        pad = int(0.08 * sr)
        seg = os.path.join(
            save_dir, speaker_names[label], "dev", "segment")
        os.makedirs(os.path.join(
            seg), exist_ok=True)
        count = 0
        for i in range(0, len(wav) - step, step):
            segment = wav[i:i + step]
            # ディレクトリの生成
            save_dir1 = os.path.join(seg, os.path.basename(
                full_path).replace(".wav", ""))
            os.makedirs(save_dir1, exist_ok=True)

            # ファイル名生成（例: 63_0.wav, 63_1.wav, ...）
            filename = f"{os.path.basename(full_path).replace('.wav', '')}_{count}.wav"
            save_file = os.path.join(save_dir1, filename)

            # 書き込み
            sf.write(save_file, segment, SAMPLING_RATE)
            # segment = np.pad(segment, (0, pad), mode='constant')
            features = (One_xvector(segment, wav_exits=True))
            X_test.append(features)
            y_test.append(label)

test_list = {}
# 0.18 から 3.0 を100等分したリストを作成
# a = np.linspace(0.2, 1.0, 10).tolist() + [1.0, 1.5, 2.0, 2.5, 3.0]
a = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
a.reverse()
for test_segment_len in a:
    X_test_, y_test_ = [], []
    for label in test_files.keys():
        saved_wav_count = 0
        for full_path in test_files[label]:

            file_path = os.path.join(full_path)
            wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
            wav = wav.numpy()

          #  wav, _ = remove_silence_with_vad(wav)
            re = os.path.join(
                save_dir, speaker_names[label], "test")
            save_file = os.path.join(
                re, f"{os.path.basename(full_path)}.wav")
            print(save_file)
            os.makedirs(os.path.join(
                re), exist_ok=True)
            sf.write(save_file,  wav,  SAMPLING_RATE)
            saved_wav_count += 1

            if len(wav) == 0 or len(wav) <= 1600*3:
                continue

            step = int(test_segment_len * SAMPLING_RATE)
            for i in range(0, len(wav) - step, step):
                segment = wav[i:i + step]
                features = One_xvector(segment, wav_exits=True)
                X_test_.append(features)
                y_test_.append(label)
    test_list[test_segment_len] = {
        "X_test":  np.array(X_test_), "y_test": np.array(y_test_)}


# PyTorch の DNN モデル定義


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # スケーリング因子
        self.m = m  # マージン
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(math.pi) - m)
        self.mm = torch.sin(torch.tensor(math.pi) - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input),
                          F.normalize(self.weight))  # cos(θ)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = torch.softmax(
            q @ k.transpose(-2, -1) / (x.size(-1) ** 0.5), dim=-1)
        return attention @ v

# PyTorch の DNN モデル定義


class SpeakerDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SpeakerDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.feature_dim = 128
        self.arc_margin = ArcMarginProduct(self.feature_dim, num_classes)
        self.activation = Swish()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, label=None):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.activation(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        if label is not None:
            x = self.arc_margin(x, label)
        return x


def inference(embedding, arc_margin):
    """
    embedding: 特徴量ベクトル (batch_size, embedding_dim)
    arc_margin: ArcMarginProduct の weight を持つ層
    """
    # Normalize input and weight
    normalized_input = F.normalize(embedding)
    normalized_weight = F.normalize(arc_margin.weight)

    # cosine similarity
    # shape: (batch_size, num_classes)
    cosine = F.linear(normalized_input, normalized_weight)
    return cosine


# データの前処理

# Save the scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
weights = class_weights[y_train]
print(f"weights:{weights}")

sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# モデル・損失関数・最適化手法の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 12  # len(set(y_train))  # クラス数
model = SpeakerDNN(
    input_dim=X_train.shape[1], num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-6)

# 学習ループ
epochs = 50000
early_stopping_patience = 1000
best_accuracy = 0
patience_counter = 0
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"current_time:{current_time}")
save_dir = "/work/asano.yuki/model/demo/話者認識/2025_04_16/model_save"
joblib.dump(scaler, os.path.join(save_dir, f"scaler_{current_time}.pkl"))

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels)  # ← label を渡す
    #    print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()

    # 検証（label なし）
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            embeddings = model(inputs)  # ラベルを渡さない場合、特徴だけ返す
            logits = inference(embeddings, model.arc_margin)  # 上記の関数を使う
          #  print(logits)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(
        f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}")

    # 早期停止
    # 早期停止
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        patience_counter = 0
        model_filename = os.path.join(
            save_dir, f"best_model_{current_time}.pth")
        torch.save(model.state_dict(), model_filename)  # モデル保存
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# 最良モデルをロードして評価
latest_model_filename = os.path.join(
    save_dir, f"best_model_{current_time}.pth")
model.load_state_dict(torch.load(latest_model_filename))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        embeddings = model(inputs)  # ラベルを渡さない場合、特徴だけ返す
        logits = inference(embeddings, model.arc_margin)  # 上記の関数を使う
        _, preds = torch.max(logits, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print("Final Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# 評価関数を定義


def evaluate_model(X_test, y_test, model):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = model(inputs)  # ラベルを渡さない場合、特徴だけ返す
            logits = inference(embeddings, model.arc_margin)  # 上記の関数を使う
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds), all_labels, all_preds


# テストセグメントの長さごとに評価
for test_segment_len in a:
    X_test = scaler.transform(test_list[test_segment_len]["X_test"])
    y_test = test_list[test_segment_len]["y_test"]

    accuracy, all_labels, all_preds = evaluate_model(X_test, y_test, model)
    print(f"Test Segment {test_segment_len}s Accuracy: {accuracy * 100:.2f}%")
    # print("Final Test Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

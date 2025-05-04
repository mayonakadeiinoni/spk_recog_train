#!/usr/bin/env python
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
# from transformers.pipelines.audio_utils import VADIterator
SAMPLING_RATE = 16000
global vad
vad = load_silero_vad()
random.seed(42)


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


# 音声データが格納されているディレクトリ
base_dirs = ["/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/名工大/高木ToWav",
             "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/名工大/山下ToWav",
             "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/名工大/柏木ToWav"]

# 各ディレクトリから共通するwavファイル名を取得
# 最初のrandom.sampleに対してシードを設定
random.seed(42)

file_sets = [set(os.listdir(d)) for d in base_dirs]
common_files = sorted(set.intersection(*file_sets))

# シャッフルして 35% を test に
test_size = int(len(common_files) * 0.35)
test_files = random.sample(common_files, test_size)

# シードをリセット（ここでシード設定を解除）
# random.seed(None)

train_files = list(set(common_files) - set(test_files))

print(f"Train: {len(train_files)}, Test: {len(test_files)}")
# 各ディレクトリのラベル（m1055 -> 0, m1077 -> 1, m1089 -> 2）
label_map = {j: i for i, j in enumerate(base_dirs)}

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


# データセット作成
X_train, y_train = [], []

target_wav_count = 1000  # 各話者のターゲットWAV数
print(f"1話者当たりのデータ拡張の割合:{target_wav_count}")

train_segment_len = 0.2
min_length = max((train_segment_len/2) * SAMPLING_RATE,
                 (0.192 * SAMPLING_RATE))
print(f"min_length:{min_length}")

# start_length= int(0.1 * SAMPLING_RATE)
max_length = int(5.0 * SAMPLING_RATE)
print(f"max_length:{max_length}")
print(f"train字のsegmentのlength:{train_segment_len}")
test_segment_len = 0.2
print(f"test字のsegmentのlength:{test_segment_len}")
segment_num_area = [1000, 5000]
print(f"各WAVから抽出するセグメント数:{segment_num_area}")
segment_concate_area = [2, 200]
print(f"ランダムなセグメント数:{segment_concate_area}")
X_train, y_train = [], []


for directory in base_dirs:
    print(directory)
    speaker_segments = []  # 話者ごとのセグメント

    # すべてのWAVファイルを対象にランダムなセグメントを取得
    for file_name in train_files:
        file_path = os.path.join(directory, file_name)
        wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
        wav, _ = remove_silence_with_vad(wav)

        # ランダムなセグメントを抽出
        num_segments = random.randint(
            segment_num_area[0], segment_num_area[1])  # 各WAVから抽出するセグメント数
        for _ in range(num_segments):
            start = random.randint(0, max(0, len(wav) - min_length))
            length = random.randint(
                min_length, min(max_length, len(wav) - start))
            segment = wav[start:start + length]
            speaker_segments.append(segment)

    # 1000個の新しいWAVを作成
    new_speaker_segments = []
    saved_wav_count = 0
    save_sample_count = max(20, target_wav_count // 10)  # min(10%, 20個)
    save_dir = "/work/asano.yuki/model/demo/話者認識/xvector_voice_statistics/第13回/wav_save"
    save_path = os.path.join(
        save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
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
            y_train.append(label_map[directory])
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

X_test, y_test = [], []
for file_name in test_files:
    for directory in base_dirs:
        file_path = os.path.join(directory, file_name)

        sr = SAMPLING_RATE
        wav = read_audio(file_path, sampling_rate=SAMPLING_RATE)
        wav, _ = remove_silence_with_vad(wav)
        step = int(test_segment_len * sr)
        pad = int(0.08 * sr)
        for i in range(0, len(wav) - step, step):
            segment = wav[i:i + step]
            # segment = np.pad(segment, (0, pad), mode='constant')
            features = (One_xvector(segment, wav_exits=True))
            X_test.append(features)
            y_test.append(label_map[directory])


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"X_train:{len(X_train)},y_test:{len(y_test)}")

# クラスごとのカウント（trainデータ）
unique_train, counts_train = np.unique(y_train, return_counts=True)
train_class_distribution = dict(zip(unique_train, counts_train))

# クラスごとのカウント（testデータ）
unique_test, counts_test = np.unique(y_test, return_counts=True)
test_class_distribution = dict(zip(unique_test, counts_test))

# 結果を表示
print(
    f"X_train: {len(X_train)}, y_train class distribution: {train_class_distribution}")
print(
    f"X_test: {len(X_test)}, y_test class distribution: {test_class_distribution}")
# 訓練データとテストデータに分割（10個をtrain、残りをtest）
train_size = 10 * len(base_dirs)  # 各ラベル10個ずつ
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
# print(len(X))
# print(len(X_train))
# print(y_test)
# SVC のハイパーパラメータ調整
"""
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100, 1000],
    'svc__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10],
    'svc__kernel': ['rbf', 'poly', 'sigmoid']
}


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf'))
])
print("ハイパーパラメータ調節！！！")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=0)
grid_search.fit(X_train, y_train)

# 最適モデルでテスト
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy)
"""
# パイプラインを構築（ハイパーパラメータ調整なし）
"""
pipeline = Pipeline([
    ('scaler', StandardScaler()),
  ('mlp', MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
                           alpha=0.0001, batch_size='auto', learning_rate='adaptive', max_iter=200, random_state=42))
]
                    )
"""
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SGDClassifier(
        loss='log_loss',         # ロジスティック回帰 (多クラス分類)
        penalty='l2',            # L2 正則化
        alpha=0.0001,            # 正則化パラメータ（大きすぎるとアンダーフィット）
        learning_rate='adaptive',  # 自動調整学習率
        eta0=0.001,              # 初期学習率（適度な値）
        max_iter=1000,           # エポック数
        early_stopping=True,     # 早期停止
        validation_fraction=0.1,  # 早期停止のための検証データ割合
        class_weight='balanced',  # クラス不均衡対応
        random_state=42          # 再現性確保
    ))
])


# SMOTEを適用
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# モデルを学習
pipeline.fit(X_train, y_train)

# テストデータで予測
y_pred = pipeline.predict(X_test)

# 精度評価
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

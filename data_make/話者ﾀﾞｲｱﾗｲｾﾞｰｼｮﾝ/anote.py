#!/usr/bin/env python
from pyannote.audio import Pipeline
import torchaudio
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# モデルロード
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="")

# GPU確認（おまけ）
print("CUDA available:", torch.cuda.is_available())
print("Using GPU:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 入力ファイル
input_wav = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/高木/wav/高木真人：おしゃべり.wav"
output_wav_base = os.path.join(os.path.dirname(input_wav), "output_chunks")
os.makedirs(output_wav_base, exist_ok=True)

# 1. 話者ダイアリゼーション
diarization = pipeline(input_wav)

# 2. 話者ごとの発話時間を集計
speaker_durations = defaultdict(float)
segments_by_speaker = defaultdict(list)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    duration = turn.end - turn.start
    speaker_durations[speaker] += duration
    segments_by_speaker[speaker].append((turn.start, turn.end))

# 3. 一番話している話者を取得
most_speaking_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
print(f"Most speaking speaker: {most_speaking_speaker}")

# 4. 音声読み込み（torchaudio）
waveform, sample_rate = torchaudio.load(input_wav)

# 5. 対象話者の音声区間だけ抽出
selected_segments = segments_by_speaker[most_speaking_speaker]
extracted_wave = []

for start, end in selected_segments:
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    extracted_wave.append(waveform[:, start_sample:end_sample])

# 6. WAVファイルとして保存（全体）
combined_wave = torch.cat(extracted_wave, dim=1)

# 7. 3秒ごとに分割して保存
chunk_length = 3 * sample_rate  # 3秒
total_length = combined_wave.size(1)
num_chunks = (total_length + chunk_length - 1) // chunk_length

for i in range(num_chunks):
    start = i * chunk_length
    end = min((i + 1) * chunk_length, total_length)
    chunk = combined_wave[:, start:end]
    chunk_filename = os.path.join(output_wav_base, f"{i+1}.wav")
    torchaudio.save(chunk_filename, chunk, sample_rate)
    print(f"Saved chunk: {chunk_filename}")

# 8. ビジュアライズ（matplotlib）
fig, ax = plt.subplots(figsize=(12, 2))
y = 0
colors = {}

for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker not in colors:
        colors[speaker] = f"C{len(colors)}"
    ax.plot([turn.start, turn.end], [y, y], lw=6,
            color=colors[speaker], label=speaker)

ax.set_yticks([])
ax.set_xlabel("Time (s)")
ax.set_title("Speaker Diarization")
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
plt.tight_layout()
plt.savefig("diarization_timeline1.png")
print("Saved diarization timeline to diarization_timeline1.png")

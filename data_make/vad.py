from silero_vad import load_silero_vad
import os
import numpy as np
import soundfile as sf
from silero_vad import get_speech_timestamps, collect_chunks
import torch


def vad_trim_wav_directory(input_dir, output_dir, vad_model, target_sr=16000):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)

                # 元の構造を保った出力先パスを構成
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 音声読み込み
                import librosa

                wav, sr = librosa.load(input_path, sr=None)

                # ステレオ → モノラル
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)

                # リサンプリング（VADモデルが16kHz前提なので）
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                    # VAD 実行
                vad_output = get_speech_timestamps(
                    wav,
                    vad_model,
                    sampling_rate=target_sr,
                    threshold=0.65,
                    min_speech_duration_ms=1,
                    min_silence_duration_ms=1,
                    window_size_samples=int(target_sr * 0.1)  # 100ms
                )

                if not vad_output:
                    print(f"❌ 音声が検出されなかった: {input_path}")
                    continue
                print(vad_output)
                # 音声部分を抽出して連結
                vad_wav = collect_chunks(vad_output, torch.from_numpy(wav))

                # 書き出し
                sf.write(output_path, vad_wav, target_sr)
                print(f"✅ {input_path} → {output_path}")


# === 使用例 ===

# VADモデルの読み込み
vad_model = load_silero_vad()

input_directory = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ"
output_directory = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ/vad抽出"

vad_trim_wav_directory(input_directory, output_directory, vad_model)

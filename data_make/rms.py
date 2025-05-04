import os
import numpy as np
import soundfile as sf


def normalize_rms(wav, target_rms=0.1):
    rms = np.sqrt(np.mean(wav**2))
    return wav * (target_rms / (rms + 1e-6))


def normalize_wav_directory(input_dir, output_dir, target_rms=0.1):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)

                # 元の構造を保った出力先パスを構成
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 音声読み込み & 正規化
                wav, sr = sf.read(input_path)
                wav_normalized = normalize_rms(wav, target_rms=target_rms)

                # 書き出し
                sf.write(output_path, wav_normalized, sr)
                print(f"✅ {input_path} → {output_path}")


# === 使用例 ===
input_directory = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ/vad抽出"
output_directory = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ/vad抽出/rms正規化"
normalize_wav_directory(input_directory, output_directory, target_rms=0.1)

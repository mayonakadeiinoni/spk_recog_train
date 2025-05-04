import soundfile as sf
import os
import numpy as np
import scipy.signal
import librosa


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
    for gain_db in range(-15,15,1):
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


def save_augmented_samples(wav_path, out_dir="/work/asano.yuki/model/demo/話者認識/2025_04_16/data_make/data_aug"):
    sr = 16000
    wav, _ = librosa.load(wav_path, sr=sr)
    aug_list = augment_audio_for_speaker(wav, sr)

    os.makedirs(out_dir, exist_ok=True)
    for i, aug in enumerate(aug_list):
        filename = os.path.join(out_dir, f"aug_{i}.wav")
        sf.write(filename, aug, sr)
    print(f"{len(aug_list)}個の拡張WAVを保存しました -> {out_dir}/")


# 使用例
save_augmented_samples(
    "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ/vad抽出/rms正規化/高木/dev/25.wav")

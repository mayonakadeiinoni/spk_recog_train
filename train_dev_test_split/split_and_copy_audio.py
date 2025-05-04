import os
import random
import shutil
from pathlib import Path
random.seed(50)
# === 設定 ===
source_base = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大"  # 元のデータディレクトリ
target_base = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/学習データ"  # 出力先ベースディレクトリ
names = ["中川", "今村","佐藤","山下","川崎","柏木","浅野","渡邊","神谷","苅谷","薫田","高木"]  # 人名リスト
split_ratio = (0.8, 0.1, 0.1)  # train:dev:test

# === 処理開始 ===
for name in names:
    wav_dir = Path(source_base) / name / "wav"
    if not wav_dir.exists():
        print(f"❌ {wav_dir} が存在しません")
        continue
    wav_files=[]
    for current_dir,sub_dirs,files  in os.walk(wav_dir):
        print(current_dir)
        wav_files.extend(list(Path(current_dir).glob("*.wav")))
    total = len(wav_files)
    print(total)
    if total == 0:
        print(f"⚠️ {name} に .wav ファイルが見つかりませんでした")
        continue
    wav_files = sorted(wav_files)  # 一度ソートしてから
    random.shuffle(wav_files)  # ランダムにシャッフル

    n_train = int(total * split_ratio[0])
    n_dev = int(total * split_ratio[1])
    n_test = total - n_train - n_dev  # 残り全部を test に
    splits = {
        "train": wav_files[:n_train],
        "dev": wav_files[n_train:n_train + n_dev],
        "test": wav_files[n_train + n_dev:]
    }

    print(f"\n👤 {name} のファイル数: {total} 件")
    for split_name, files in splits.items():
        out_dir = Path(target_base) / name / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(f, out_dir / f.name)

        print(f"  📂 {split_name}: {len(files)} 件 -> {out_dir}")

print("\n✅ 全データの分割とコピーが完了しました。")

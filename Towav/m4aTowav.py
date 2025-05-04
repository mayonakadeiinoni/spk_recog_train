import os
from pydub import AudioSegment

def convert_m4a_to_wav(input_dir, output_dir):
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 指定ディレクトリ内のすべてのm4aファイルを処理
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".m4a"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".wav")
            
            # m4aをwavに変換
            audio = AudioSegment.from_file(input_path, format="m4a")
            audio.export(output_path, format="wav")
            print(f"Converted: {file_name} -> {output_path}")

# 使用例
input_directory = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/今村"
# 変換元のディレクトリを指定
output_directory = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大/今村/wav"  # 変換後のディレクトリを指定
convert_m4a_to_wav(input_directory, output_directory)

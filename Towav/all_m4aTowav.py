import os
from pydub import AudioSegment

def convert_all_m4a_to_wav(root_dir):
    for current_dir,sub_dirs,files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith(".m4a"):
                input_path = os.path.join(current_dir,file_name)
                
                output_dir = os.path.join(current_dir,"wav")
                os.makedirs(output_dir,exist_ok=True)
                
                output_path= os.path.join(output_dir,os.path.splitext(file_name)[0]+".wav")
                
                audio = AudioSegment.from_file(input_path, format="m4a")
                audio.export(output_path, format="wav")
                print(f"Converted: {input_path} -> {output_path}")
                
        

root_directory = "/work/asano.yuki/model/demo/話者認識/2025_04_16/名工大"
convert_all_m4a_to_wav(root_directory)

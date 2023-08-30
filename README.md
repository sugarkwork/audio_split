# audio_split

split_audio_on_silence で音声ファイルを分割し、whisper(large-v2) で文字起こしをして、それをファイル名にします。

例：
入力（これは（１秒）テストですという音声データだった場合）
input/hoge.wav

出力
output_0/hoge/001_これは.wav
output_0/hoge/002_テストです.wav


# install (windows, CUDA 11.8 の場合)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/whisper.git 


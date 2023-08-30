# audio_split

split_audio_on_silence で音声ファイルを分割し、whisper(large-v2) で文字起こしをして、それをファイル名にします。
誰か作ってるかもしれない。

- 例：
入力（これは（１秒無音）テストです、という音声データだった場合）
```
input/hoge.wav
```

- 出力
```
output_0/hoge/01_これは.wav
output_0/hoge/02_テストです.wav
```

# install (windows, CUDA 11.8 の場合)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/whisper.git 
```

# このコードの設定

- device="cuda:0"
  - GPU を使う場合に指定。複数の GPU がある場合は 0 じゃなくて 1 とかにする。cpu 使う場合は device="cpu" を指定する。

|設定|デフォルト値|説明|
|---|---|---|
|min_silence_len|1000|何ミリ秒、無音時間があったら音声ファイルを切るかの時間をミリ秒で指定します|
|silence_thresh|-55|何 dBFS の音以下だった場合を無音と判定するかを指定します|
|keep_silence|300|音声ファイルを切断する場合に、切断しないで含める無音の時間を指定します|

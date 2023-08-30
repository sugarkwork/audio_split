import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper


def load_whisper_model(model="large-v2", device="cuda:0"):
    return whisper.load_model(model, device=device)


def split_audio_on_silence(file_path, min_silence_len=1000, silence_thresh=-55, keep_silence=300):
    """Split an audio file based on silence and return the chunks."""
    audio_segment = AudioSegment.from_wav(file_path)
    return split_on_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh,
                            keep_silence=keep_silence)


def get_sanitized_filename(text, length=32):
    """Return a sanitized version of text suitable for filenames."""
    chars_to_replace = '\\/:?."<>|*'
    translation_table = str.maketrans({char: "_" for char in chars_to_replace})
    return text.translate(translation_table)[:length]


def save_chunks(chunks, target_dir, model=None):
    """Save audio chunks to the target directory. If model is provided, rename files based on transcribed text."""
    num_digits = len(str(len(chunks))) + 1
    for i, chunk in enumerate(chunks):
        file_number = str(i).zfill(num_digits)
        chunk_path = os.path.join(target_dir, f"{file_number}_.wav")
        chunk.export(chunk_path, format="wav")

        if model:
            result = model.transcribe(chunk_path, verbose=True, language='ja',
                                      fp16=True)  # Specify False when using the CPU.
            text = get_sanitized_filename(result['text'])

            text_path = os.path.join(target_dir, f"{file_number}_{text}.wav")
            print(f"rename: {chunk_path} -> {text_path}")
            os.rename(chunk_path, text_path)


def process_audio_files(input_dir, output_dir, model=None, min_silence_len=1000, silence_thresh=-55, keep_silence=300):
    """Process audio files from input_dir and save the split chunks to output_dir."""
    for file_name in os.listdir(input_dir):
        if not file_name.endswith('.wav'):
            print(f"skip: {file_name}")
            continue

        base_name = os.path.splitext(file_name)[0]
        print(base_name)
        file_path = os.path.join(input_dir, file_name)

        chunks = split_audio_on_silence(file_path,
                                        min_silence_len=min_silence_len,
                                        silence_thresh=silence_thresh,
                                        keep_silence=keep_silence)

        target_dir = os.path.join(output_dir, base_name)
        os.makedirs(target_dir, exist_ok=True)

        save_chunks(chunks, target_dir, model)


def main():
    whisper_model = load_whisper_model(device="cuda:0")

    output_dir = 'output'
    for n in range(100):
        output_dir = f'output_{n}'
        if not os.path.exists(output_dir):
            break

    process_audio_files(input_dir='input', output_dir=output_dir,
                        model=whisper_model,
                        min_silence_len=1000,  # Minimum silence duration in milliseconds
                        silence_thresh=-55,    # Volume threshold in dBFS for silence detection
                        keep_silence=300)      # Amount of silence to keep before and after the actual audio


if __name__ == "__main__":
    main()

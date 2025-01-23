import whisper
import os
import json
import torchaudio
import argparse
import torch
from collections import deque

lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
}

def transcribe_one(audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return lang, result.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="medium")
    parser.add_argument("--last_file", type=str, help="Path to the last saved annotation file (optional)", default=None)
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "[ZH]",
        }
    assert (torch.cuda.is_available()), "Please enable GPU in order to run Whisper!"
    model = whisper.load_model(args.whisper_size)
    parent_dir = "./custom_character_voice/"
    speaker_annos = []
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # resample audios
    # 2023/4/21: Get the target sampling rate
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    processed_files = 0
    speaker = list(os.walk(parent_dir))[0][1][0]
    parent_dir += speaker

    # 初始化保存文件的队列
    saved_files_queue = deque(maxlen=4)  # 最多保存4个文件
    save_interval = 100  # 每100次循环保存一次

    # 如果提供了上次的文件，读取上次的文件内容
    if args.last_file and os.path.exists(args.last_file):
        with open(args.last_file, 'r', encoding='utf-8') as f:
            last_saved_lines = f.readlines()
        last_saved_files = [line.split("|")[0] for line in last_saved_lines]
        print(f"Resuming from last saved file: {args.last_file}")
    else:
        last_saved_files = []
        print("No last saved file provided. Starting from scratch.")

    for i, wavfile in enumerate(list(os.walk(parent_dir))[0][2]):
        # try to load file as audio
        if wavfile.startswith("processed_"):
            continue
        wavfile_path = os.path.join(parent_dir, wavfile)
        if wavfile_path in last_saved_files:
            print(f"Skipping already processed file: {wavfile}")
            continue
        try:
            wav, sr = torchaudio.load(wavfile_path, frame_offset=0, num_frames=-1, normalize=True,
                                      channels_first=True)
            wav = wav.mean(dim=0).unsqueeze(0)
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
            if wav.shape[1] / sr > 20:
                print(f"{wavfile} too long, ignoring\n")
            # transcribe text
            lang, text = transcribe_one(wavfile_path)
            if lang not in lang2token:
                print(f"{lang} not supported, ignoring\n")
                continue
            text = lang2token[lang] + text + lang2token[lang] + "\n"
            speaker_annos.append(wavfile_path + "|" + text)
            processed_files += 1
            print(f"Processed: {processed_files}/{total_files}")

            # 每100次循环保存一次文件
            if processed_files % save_interval == 0:
                save_file_path = f"short_character_anno_{processed_files}.txt"
                with open(save_file_path, 'w', encoding='utf-8') as f:
                    for line in speaker_annos:
                        f.write(line)
                print(f"Saved annotation to {save_file_path}")
                speaker_annos.clear()  # 清空已保存的注释

                # 将当前保存的文件路径加入队列
                saved_files_queue.append(save_file_path)

                # 如果队列已满，删除最旧的文件
                if len(saved_files_queue) == saved_files_queue.maxlen:
                    oldest_file = saved_files_queue[0]
                    print(f"Deleting oldest file: {oldest_file}")
                    os.remove(oldest_file)

        except Exception as e:
            print(f"Error processing file {wavfile}: {e}")
            continue

    # 如果最后还有未保存的内容，保存到文件
    if speaker_annos:
        save_file_path = f"short_character_anno_{processed_files}.txt"
        with open(save_file_path, 'w', encoding='utf-8') as f:
            for line in speaker_annos:
                f.write(line)
        print(f"Saved remaining annotations to {save_file_path}")
        speaker_annos.clear()

    if len(saved_files_queue) == saved_files_queue.maxlen:
        oldest_file = saved_files_queue[0]
        print(f"Deleting oldest file: {oldest_file}")
        os.remove(oldest_file)

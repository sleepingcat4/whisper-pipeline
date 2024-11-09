import whisper
import torch
from pydub import AudioSegment
import os
import tempfile
import csv
import time

def timestamp(ms):
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def slice_audio(audio, start_ms, end_ms, temp_folder, chunk_id):
    slice_audio = audio[start_ms:end_ms]
    chunk_file_path = os.path.join(temp_folder, f"chunk_{chunk_id}.wav")
    slice_audio.export(chunk_file_path, format="wav")
    return chunk_file_path

def transcribe_chunk(model, chunk_file_path):
    result = model.transcribe(chunk_file_path)
    return result["text"]

def process_audio_file(input_file, model):
    audio = AudioSegment.from_wav(input_file)
    audio_duration = len(audio)
    
    transcriptions = []

    with tempfile.TemporaryDirectory() as temp_folder:
        for start_ms in range(0, audio_duration, 30000):
            end_ms = min(start_ms + 30000, audio_duration)
            chunk_file_path = slice_audio(audio, start_ms, end_ms, temp_folder, start_ms)
            result_text = transcribe_chunk(model, chunk_file_path)
            transcriptions.append([timestamp(start_ms), timestamp(end_ms), result_text])

    return transcriptions

def save_transcription_to_csv(audio_file, transcriptions, output_folder):
    file_name = os.path.basename(audio_file).replace(".wav", ".csv")
    csv_file = os.path.join(output_folder, file_name)
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Start Time", "End Time", "Transcription"])
        writer.writerows(transcriptions)

def process_folder(folder_path, output_folder, model):
    start_time = time.time()
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            input_file = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            transcriptions = process_audio_file(input_file, model)
            save_transcription_to_csv(input_file, transcriptions, output_folder)
            print(f"Transcription saved for {file_name}")
    
    end_time = time.time()
    total_time = end_time - start_time
    total_hours = total_time / 3600
    return total_hours

def main():
    input_folder = input("Please provide the path to your folder containing audio files: ")
    output_folder = input("Please provide the folder where you want to save the transcription CSV files: ")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("turbo", device=device)
    
    show_duration = input("Do you want to know how many hours it took to process the folder? (yes/no): ").strip().lower()
    
    total_hours = process_folder(input_folder, output_folder, model)
    
    if show_duration == "yes":
        print(f"\nTotal time taken: {total_hours:.2f} hours")

if __name__ == "__main__":
    main()

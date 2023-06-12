import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import soundfile as sf

# Define the path to the directory with your .mp3 files
directory_path = "./audio"

# Load pre-trained model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Define the chunk length in seconds
chunk_length = 30  # 30 seconds

# Iterate over all .mp3 files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".mp3"):
        print(f"Processing {filename}")

        # Convert .mp3 file to .wav
        audio = AudioSegment.from_mp3(os.path.join(directory_path, filename))
        audio.export("temp.wav", format="wav")

        # Load audio
        audio_input, _ = sf.read("temp.wav")

        # Transcribe in chunks
        transcription = ""
        for i in range(0, len(audio_input), chunk_length * 16000):  # 16000 samples/second
            chunk = audio_input[i:i + chunk_length * 16000]

            # Preprocess audio
            input_values = processor(chunk, sampling_rate=16_000, return_tensors="pt").input_values

            # Perform transcription
            with torch.no_grad():
                logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode the audio to generate transcript
            chunk_transcription = processor.decode(predicted_ids[0])
            transcription += chunk_transcription + " "

        print(transcription)

        # Save the transcription to a .txt file
        with open(os.path.splitext(filename)[0] + ".txt", "w") as txt_file:
            txt_file.write(transcription)

# Remember to delete the temporary .wav file
os.remove("temp.wav")

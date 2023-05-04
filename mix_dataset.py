import os
import random
import shutil
import librosa
import pydub
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from pydub.effects import normalize
from tqdm import tqdm
# Set the input and output directories
speaker_dir = '/home/amr/Desktop/Gatech/Spring2023/DL/Project/TriAAN-VC/Data/wav48_silence_trimmed'
urban_dir = '/home/amr/sound_datasets/urbansound8k/audio'
mix_dir = '/home/amr/Desktop/Gatech/Spring2023/DL/Project/TriAAN-VC/Data/Mixed/mixed'
utt_dir = '/home/amr/Desktop/Gatech/Spring2023/DL/Project/TriAAN-VC/Data/Mixed/utterances'
urb_dir = '/home/amr/Desktop/Gatech/Spring2023/DL/Project/TriAAN-VC/Data/Mixed/urban'
meta_dir = '/home/amr/Desktop/Gatech/Spring2023/DL/Project/TriAAN-VC/Data/txt'
txt_dir = '/home/amr/Desktop/Gatech/Spring2023/DL/Project/TriAAN-VC/Data/Mixed/txt'

# Create the output directories if they don't exist
os.makedirs(mix_dir, exist_ok=True)
os.makedirs(utt_dir, exist_ok=True)
os.makedirs(urb_dir, exist_ok=True)

urb_vol_factor = 0.2


# Load the urban sound files
urb_files = []
for dirpath, dirnames, filenames in os.walk(urban_dir):
    for filename in filenames:
        if filename.endswith('.wav'):
            urb_files.append(os.path.join(dirpath, filename))

# Load the speaker utterance files and mix them with random urban sound files
past = ''
for speaker_folder in tqdm(sorted(os.listdir(speaker_dir))[:12]):
    os.makedirs(os.path.join(mix_dir, speaker_folder), exist_ok=True)
    os.makedirs(os.path.join(utt_dir, speaker_folder), exist_ok=True)
    os.makedirs(os.path.join(urb_dir, speaker_folder), exist_ok=True)
    os.makedirs(os.path.join(txt_dir, speaker_folder), exist_ok=True)
    if not os.path.isdir(os.path.join(speaker_dir, speaker_folder)):
        continue
    for index, utterance_file in tqdm(enumerate(sorted(os.listdir(os.path.join(speaker_dir, speaker_folder))))):
        if not utterance_file.endswith('.flac'):
            continue
        utterance_name = os.path.splitext(utterance_file)[0]
        txt_file_name = '_'.join(os.path.splitext(utterance_file)[0].split('_')[:-1])
        txt_file_path = os.path.join(meta_dir,speaker_folder, txt_file_name+'.txt')
        # Load the speaker utterance file and convert it to a temporary WAV file
        utterance_path = os.path.join(speaker_dir, speaker_folder, utterance_file)
        utterance_data, utterance_sr = sf.read(utterance_path)
        utterance_temp_path = os.path.join(utt_dir, os.path.splitext(utterance_file)[0] + '.wav')
        sf.write(utterance_temp_path, utterance_data, utterance_sr, format='WAV', subtype='PCM_16')

        # Load a random urban sound file and convert it to a temporary WAV file with reduced volume
        urb_path = random.choice(urb_files)
        urb_data, urb_sr = sf.read(urb_path)
        
        # Trim the urban sound to match the length of the speaker utterance, if necessary
        urb_data = urb_data[:len(utterance_data)]
        urb_data = urb_vol_factor * urb_data / np.max(np.abs(urb_data))
        
        urb_temp_path = os.path.join(urb_dir, os.path.basename(urb_path).split('.')[0] + '.wav')
        sf.write(urb_temp_path, urb_data, urb_sr, format='WAV', subtype='PCM_16')

        # Load the speaker utterance and the urban sound and mix them
        if len(urb_data.shape)>1:
            urb_data = np.mean(urb_data,axis=1)
        mixed_data = utterance_data.copy()
        mixed_data[:len(urb_data)] = mixed_data[:len(urb_data)] + urb_data
        mixed_sr = utterance_sr
        
        # Save the mixed audio file to the output directory
        mixed_path = os.path.join(mix_dir, speaker_folder, '{}.wav'.format(utterance_name))
        sf.write(mixed_path, mixed_data, mixed_sr, format='WAV', subtype='PCM_16')

        # Delete the temporary files
        os.remove(utterance_temp_path)
        os.remove(urb_temp_path)

        # Copy the speaker utterance file to the output directory
        shutil.copy(utterance_path, os.path.join(utt_dir, speaker_folder, '{}.flac'.format(utterance_name)))
        # shutil.copy(urb_path, os.path.join(urb_dir,  speaker_folder, '{}.wav'.format(utterance_name)))
        sf.write(os.path.join(urb_dir,  speaker_folder, '{}.wav'.format(utterance_name)), 
                 urb_data, urb_sr, format='WAV', subtype='PCM_16')

        if not os.path.exists(os.path.join(txt_dir, speaker_folder, '{}.txt'.format(txt_file_name))):
            shutil.copy(txt_file_path, os.path.join(txt_dir, speaker_folder, '{}.txt'.format(txt_file_name)))

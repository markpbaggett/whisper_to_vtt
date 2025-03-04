import argparse
import warnings
# @Todo: remove this when numba is updated to 0.53.1
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import os
from tqdm import tqdm
import whisper
from whisper.utils import get_writer
import time
import logging

# Configure logging
logging.basicConfig(
    filename="default.log",  # Log file name
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)


class Transcriber:
    def __init__(self, directory, output, language_model, language, fp16, ignore_existing=False):
        self.directory = directory
        self.files_to_transcribe = (".mp3", ".mp4", ".avi", ".flac", ".mkv", ".wmv", ".wma", ".wav", ".webm")
        self.output = output
        self.model = language_model
        self.language = language
        self.fp16 = fp16
        self.ignore_existing = ignore_existing
        if self.ignore_existing:
            self.existing = self.__cache_existing()
        else:
            self.existing = []

    def __cache_existing(self):
        existing = []
        for dirpath, dirnames, filenames in os.walk(self.output):
            for filename in filenames:
                if filename.endswith('.vtt'):
                    existing.append(f'{os.path.splitext(filename)[0]}')
        return existing

    def batch_transcribe(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for filename in tqdm(filenames):
                if f'{os.path.splitext(filename)[0]}_{self.model}' not in self.existing and filename.endswith(self.files_to_transcribe):
                    print(filename)
                    self.__transcribe(os.path.join(self.directory, filename))
                else:
                    print(f"Skipping {filename}:  already exists!")
        return

    def __transcribe(self, file):
        model = whisper.load_model(self.model)

        start_time = time.time()

        result = model.transcribe(file, fp16=self.fp16, language=self.language, word_timestamps=True)

        end_time = time.time()
        elapsed_time = end_time - start_time

        os.makedirs(self.output, exist_ok=True)
        output_file = f'{os.path.splitext(file)[0]}_{self.model}.vtt'
        writer = get_writer('vtt', self.output)
        writer_json = get_writer('json', self.output)
        writer(result, output_file)
        writer_json(result, output_file.replace('.vtt', '.json'))

        logging.info(f"Transcription completed for {file} in {elapsed_time:.2f} seconds.")

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='Specify directory to files.', required=True)
    parser.add_argument('-o', '--output', help='Specify output directory', default='output')
    parser.add_argument('-m', '--model', help='Specify model', default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('-l', '--language', help='Specify language', default='English')
    parser.add_argument('-f', '--fp16', action='store_true', help='Use FP16 instead of FP32')
    parser.add_argument('-i', '--ignore_existing', action='store_true', help='Ignore A/V files that already have a transcription')
    args = parser.parse_args()
    x = Transcriber(args.directory, args.output, args.model, args.language, args.fp16, args.ignore_existing)
    x.batch_transcribe()

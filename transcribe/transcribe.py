import argparse
import warnings
# @Todo: remove this when numba is updated to 0.53.1
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import os
from tqdm import tqdm
import whisper
import shutil
from whisper.utils import get_writer

class Transcriber:
    def __init__(self, directory, output, language_model, language, fp16, ignore_existing=False):
        self.directory = directory
        self.output = output
        self.model = language_model
        self.language = language
        self.fp16 = fp16
        self.ignore_existing = ignore_existing
        if self.ignore_existing:
            self.existing = self.__cache_existing()
        else:
            self.existing = []
        self.bad_directory

    def __cache_existing(self):
        existing = []
        for dirpath, dirnames, filenames in os.walk(self.output):
            for filename in filenames:
                if filename.endswith('.vtt'):
                    existing.append(f'{os.path.splitext(filename)[0]}')
        return existing

    def move_bad_file(source, destination):
        shutil.move(source, destination)
    
    def batch_transcribe(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for filename in tqdm(filenames):
                if f'{os.path.splitext(filename)[0]}_{self.model}' not in self.existing:
                    try:
                        self.__transcribe(os.path.join(self.directory, filename))
                    except Exception:
                        source_directory = 'directory/filename'
                        destination_directory = 'fail_directory'
                        move_bad_file(source_directory, destination_directory)
                        
        return

    def __transcribe(self, file):
        model = whisper.load_model(self.model)
        result = model.transcribe(file, fp16=self.fp16, language=self.language)
        os.makedirs(self.output, exist_ok=True)
        output_file = f'{os.path.splitext(file)[0]}_{self.model}.vtt'
        writer = get_writer('vtt', self.output)
        writer(result, output_file)
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
    parser.add_argument('-b', '--bad_directory', help='Gives a directory to move erring files to.', required=False)
    args = parser.parse_args()
    x = Transcriber(args.directory, args.output, args.model, args.language, args.fp16, args.ignore_existing, args.bad_directory)
    x.batch_transcribe()

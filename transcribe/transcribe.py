import argparse
import os
from tqdm import tqdm
import whisper
from whisper.utils import get_writer

class Transcriber:
    def __init__(self, directory, output, language_model, language):
        self.directory = directory
        self.output = output
        self.model = language_model
        self.language = language

    def batch_transcribe(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for filename in tqdm(filenames):
                self.__transcribe(f'{self.directory}/{filename}')

    def __transcribe(self, file):
        model = whisper.load_model(self.model)
        result = model.transcribe(file, language=self.language)
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        writer = get_writer('vtt', self.output)
        writer(result, f'{self.output}/{file.split(".")[0]}.vtt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='Specify directory to files.', required=True)
    parser.add_argument('-o', '--output', help='Specify output directory', default='output')
    parser.add_argument('-m', '--model', help='Specify model', default='base')
    parser.add_argument('-l', '--language', help='Specify language', default='English')
    args = parser.parse_args()
    x = Transcriber(args.directory, args.output, args.model, args.language)
    x.batch_transcribe()

import argparse

parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

parser.add_argument('-i', '--input_data', default='../ESDcorpus/', type=str,
                    help='Path to your LibriSpeech directory', required=False)

args = parser.parse_args()
print(args)

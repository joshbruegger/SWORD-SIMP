import gdown
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Download directory')
    args = parser.parse_args()

    if os.path.exists(args.output):
        print("fatal: output directory already exists. Already downloaded?")
        exit()

    url = "https://drive.google.com/drive/folders/1Ouga4ms22NK-sDUkI4MoFqHuhxG2qG1i"
    gdown.download_folder(url, use_cookies=False,
                          output=args.output, quiet=False)

import gdown
import os
import argparse


if __name__ == '__main__':
    print("Downloading files...")

    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Download directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help="Force the removal of the existing 'cropped' directory if it exists")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        print("Output directory already exists. Use --force to overwrite.")
        exit()

    url = "https://drive.google.com/drive/folders/1Ouga4ms22NK-sDUkI4MoFqHuhxG2qG1i"
    gdown.download_folder(url, use_cookies=False,
                          output=args.output, quiet=False)

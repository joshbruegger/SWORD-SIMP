import gdown
import os
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Download directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help="Force the removal of the existing 'cropped' directory if it exists")
    args = parser.parse_args()

    print("Downloading files...")

    if os.path.exists(args.output):
        if args.force:
            shutil.rmtree(args.output)
        else:
            print("Output directory already exists. Use --force to overwrite.")
            exit()

    url = "https://drive.google.com/drive/folders/1Ouga4ms22NK-sDUkI4MoFqHuhxG2qG1i"
    gdown.download_folder(url, use_cookies=False,
                          output=args.output, quiet=False)

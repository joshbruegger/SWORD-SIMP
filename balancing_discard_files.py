import os
import sys


def delete_file(file):
    try:
        # Check if the file exists
        if os.path.isfile(file):
            # Delete the file
            os.remove(file)
            print(f"Successfully deleted {file}")
        else:
            print(f"Error: {file} does not exist")
    except Exception as e:
        print(f"An error occurred while deleting {file}: {e}")


def delete_files(file_path):
    # Open the file in read mode
    with open(file_path, "r") as f:
        # Read lines from the file
        lines = f.readlines()

        # Loop through each line (file)
        for line in lines:
            # Strip the newline character from the line
            file_to_delete = line.strip()
            image_file_to_delete = file_to_delete.replace(
                "/labels/", "/images/"
            ).replace(".txt", ".jpg")

            # Delete the files
            delete_file(file_to_delete)
            delete_file(image_file_to_delete)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python balancing_discard_files.py <file>")
        sys.exit(1)

    delete_files(sys.argv[1])

    # TODO: Delete relevat entries from crops.txt in order to avoid having to re-run the crop script only to delete the same files again

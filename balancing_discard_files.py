import os
import sys


def delete_file(file, dest):
    try:
        if os.path.isfile(file):
            # os.remove(file)
            os.rename(file, dest)
            print(f"Successfully deleted {file}")
        else:
            print(f"Error: {file} does not exist")
    except Exception as e:
        print(f"An error occurred while deleting {file}: {e}")


def delete_files(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            file_to_delete = line.strip()
            image_file_to_delete = file_to_delete.replace(
                "/labels/", "/images/"
            ).replace(".txt", ".jpg")

            deleted_path = file_to_delete.replace("/labels/", "/deleted/labels/")
            deleted_image_path = image_file_to_delete.replace(
                "/images/", "/deleted/images/"
            )
            os.makedirs(os.path.dirname(deleted_path), exist_ok=True)
            os.makedirs(os.path.dirname(deleted_image_path), exist_ok=True)

            delete_file(file_to_delete, deleted_path)
            delete_file(image_file_to_delete, deleted_image_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python balancing_discard_files.py <file>")
        sys.exit(1)

    delete_files(sys.argv[1])

    # TODO: Delete relevat entries from crops.txt in order to avoid having to re-run the crop script only to delete the same files again

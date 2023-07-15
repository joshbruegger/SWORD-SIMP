import os
import sys

def delete_files(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as f:
        # Read lines from the file
        lines = f.readlines()

        # Loop through each line (file)
        for line in lines:
            # Strip the newline character from the line
            file_to_delete = line.strip()
            try:
                # Check if the file exists
                if os.path.isfile(file_to_delete):
                    # Delete the file
                    os.remove(file_to_delete)
                    print(f'Successfully deleted {file_to_delete}')
                else:
                    print(f'Error: {file_to_delete} does not exist')
            except Exception as e:
                print(f'An error occurred while deleting {file_to_delete}: {e}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python balancing_discard_files.py <file>")
        sys.exit(1)

    delete_files(sys.argv[1])

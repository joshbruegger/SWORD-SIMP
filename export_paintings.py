import argparse
import os
import gc
from psd_tools import PSDImage


def export_psd(filename, output_folder, max_dim):
    print(f"Processing {filename}...")

    psd = PSDImage.open(filename)
    layer_names = [layer.name for layer in psd]
    bg_names = ["Sfondo", "Background"]
    if not any(name in bg_names for name in layer_names):
        print("No background layer found. using all layers (slower)")
        image = psd.composite()
    else:
        image = psd.composite(
            layer_filter=lambda layer: layer.is_visible()
            and layer.name in ["Sfondo", "Background"]
        )

    # Get the dimensions of the PSD file
    width, height = image.size
    print(f"Dimensions: {width}x{height}")

    # Calculate the scaling factor
    scale = max_dim / max(width, height)
    print(f"Scale: {scale}")

    # Resize
    image = image.resize((int(width * scale), int(height * scale)))
    print(f"New dimensions: {image.size}")

    # Save the image
    image.save(
        os.path.join(output_folder, f"{os.path.basename(filename).split('.')[0]}.png")
    )

    del image
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Export paintings from PSD files.")
    parser.add_argument(
        "file_path", type=str, help="Path to the folder containing the PSD files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Path to the folder where the bounding box files will be saved.",
    )
    parser.add_argument(
        "--max_dim",
        type=int,
        default=1024,
        help="Maximum dimension of the exported paintings.",
    )
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for filename in os.listdir(args.file_path):
        if filename.endswith((".psd", ".psb")):  # Only process PSD files
            export_psd(
                os.path.join(args.file_path, filename), args.output, args.max_dim
            )


if __name__ == "__main__":
    main()

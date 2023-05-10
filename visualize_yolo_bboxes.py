import argparse
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_yolo_bboxes(image_path, output_path=None):
    # Check if the .txt file exists
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if not os.path.exists(txt_path):
        print(f"No .txt file found for image: {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Read the bounding boxes
    with open(txt_path, 'r') as f:
        bboxes = f.readlines()

    # Visualize the bounding boxes
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox in bboxes:
        class_id, cx, cy, w, h = map(float, bbox.strip().split())

        x = (cx - w/2) * width
        y = (cy - h/2) * height
        w = w * width
        h = h * height

        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, str(int(class_id)), fontsize=12, color='r')

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes")
    parser.add_argument("image_path", help="Path to the .png image file")
    parser.add_argument("-o", "--output", dest="output_path", nargs='?', const=True,
                        help="Path to save the visualization image or flag to save with default name", default=None)
    args = parser.parse_args()

    if args.output_path is True:
        args.output_path = os.path.join(os.path.dirname(
            args.image_path), f"visualization_{os.path.basename(args.image_path)}")

    visualize_yolo_bboxes(args.image_path, args.output_path)

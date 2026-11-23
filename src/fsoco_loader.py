import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# Locate the dataset relative to this Python file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "fsoco"


def load_fsoco_image(team="ampera", image_name="amz_00805.png"):
    image_path = DATASET_DIR / team / "img" / image_name
    annotation_path = DATASET_DIR / team / "ann" / (image_name + ".json")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Load annotation
    with open(annotation_path, "r", encoding="utf-8") as file:
        annotation = json.load(file)

    return image, annotation


def draw_bounding_boxes(image, annotation):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    colors = {
        "blue_cone": "blue",
        "yellow_cone": "yellow",
        "orange_cone": "orange",
        "large_orange_cone": "red"
    }

    for obj in annotation["objects"]:
        if obj["geometryType"] != "rectangle":
            continue

        label = obj["classTitle"]
        points = obj["points"]["exterior"]

        x1, y1 = points[0]
        x2, y2 = points[1]

        color = colors.get(label, "white")

        # Draw bounding box
        draw.rectangle(
            [min(x1, x2), min(y1, y2),
             max(x1, x2), max(y1, y2)],
            outline=color,
            width=3
        )

        # Draw label
        draw.text((x1, max(0, y1 - 15)), label, fill=color)

    return image

def extract_cone_detections(annotation):
    detections = []

    for obj in annotation["objects"]:
        if obj["geometryType"] != "rectangle":
            continue

        label = obj["classTitle"]
        points = obj["points"]["exterior"]

        x1, y1 = points[0]
        x2, y2 = points[1]

        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detections.append({
            "class": label,
            "bbox": [x1, y1, x2, y2],
            "center": [center_x, center_y]
        })

    return detections


def main():
    image, annotation = load_fsoco_image()

    # Extract numerical cone detections
    detections = extract_cone_detections(annotation)

    print(f"Found {len(detections)} cone detections:\n")

    for i, detection in enumerate(detections):
        print(
            f"Cone {i + 1}: "
            f"{detection['class']}, "
            f"Center: {detection['center']}"
        )

    # Display the annotated image
    result = draw_bounding_boxes(image, annotation)

    plt.figure(figsize=(14, 7))
    plt.imshow(result)
    plt.axis("off")
    plt.title("FSOCO Cone Bounding Boxes")
    plt.show()


if __name__ == "__main__":
    main()
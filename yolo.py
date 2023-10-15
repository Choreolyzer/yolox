import os
import argparse
import cv2
import numpy as np
import json


def inference(image_folder):
    result = {}

    # Load the model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load image files
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]

    for img_str in images:
        id = int(img_str[:-4])
        if id % 3 != 0 and id != 1:
            continue

        image = cv2.imread(os.path.join(image_folder, img_str))
        height, width, channels = np.shape(image)

        # Detect people
        blob = cv2.dnn.blobFromImage(image, 1/255., (416, 416), (0,0,0), True, crop=False)
        net.setInput(blob)
        predictions = net.forward(output_layers)

        detections = []

        # Handle predictions
        for p in predictions:
            for detection in p:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id != 0:   # must be human
                    continue
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    detections.append(f"{x},{y},{w},{h},{confidence}\n")

                    cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,0), 2)

        result[os.path.join(image_folder, f"{img_str[:-4].zfill(8)}.txt")] = detections
    
        cv2.imshow("Video", image)

    with open(os.path.join("", "det_db_motrv2.json"), "w") as file:
        json_str = json.dumps(result)
        file.write(json_str)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--images", help="path to image folder")
    # a.add_argument("--output", help="path to output folder")
    args = a.parse_args()
    print(args)
    inference(args.images)

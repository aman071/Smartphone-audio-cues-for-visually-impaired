import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

# Load the pre-trained Mask R-CNN
model = hub.load("https://tfhub.dev/tensorflow/mask_rcnn/inception_v2/1")

video_path = 'path/to/your/video/file.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video file opened
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# List to store results for each frame
all_results = []

while True:
    # Read a frame
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Convert frame to RGB (Mask R-CNN expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb_frame[np.newaxis, ...])

    # Detection happens here
    results = model(input_tensor)       

    # Extract bounding boxes and masks
    boxes = results['detection_boxes'].numpy()
    masks = results['detection_masks'].numpy()

    # List to store results for the current frame
    frame_results = []

    # Draw bounding boxes and masks on the frame
    for i in range(boxes.shape[0]):
        box = boxes[i]
        mask = masks[i]
        # Convert mask to uint8 format
        mask = (mask > 0.5).astype(np.uint8)

        # Draw bounding box
        x, y, w, h = box
        x, y, w, h = int(x * frame.shape[1]), int(y * frame.shape[0]), int(w * frame.shape[1]), int(h * frame.shape[0])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw mask
        mask_color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.5, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

        # Get the class name for the detected object
        class_id = int(results['detection_classes'][i])
        class_name = results['class_names'][class_id]

        # Store results
        object_result = {
            'class_name': class_name,
            'confidence': results['detection_scores'][i],
            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h}
        }
        frame_results.append(object_result)

        # label = f"{class_name}: {results['detection_scores'][i]:.2f}"
        # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    all_results.append(frame_results)
    cv2.imshow('Instance Segmentation', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

output_file = 'results.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f)

# print(f'Results saved to {output_file}')
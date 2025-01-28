import math
import os
import shutil
import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11l.pt")

# Open the video file
video_path = "footage/test_video9.mp4"
cap = cv2.VideoCapture(video_path)

# Define the output directory
output_dir = "runs/debug/relocking"

# Remove the directory and all its contents (including subfolders) if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Recreate the directory
os.makedirs(output_dir)


def save_annotated_frame(annotated_frame):
    # Determine the next filename based on the maximum number in the folder
    existing_files = [int(f.split('.')[0]) for f in os.listdir(output_dir) if f.split('.')[0].isdigit()]
    next_number = max(existing_files, default=0) + 1
    output_path = os.path.join(output_dir, f"{next_number}.jpg")

    # Save the annotated frame
    cv2.imwrite(output_path, annotated_frame)


def calculate_distance(frame_height, h_pixels):
    """
    Calculate the distance from the drone to a person.

    Parameters:
    H (float): Actual height of the person in meters (e.g., 1.8 for 180 cm).
    frame_height (int): Height of the camera frame in pixels.
    h_pixels (int): Height of the person in the image (bounding box height in pixels).
    alpha_degrees (float): Tilt angle of the camera in degrees.

    Returns:
    float: The distance from the drone to the person in meters.
    """
    H = 1.8  # Height of the person in meters (180 cm)
    W = 0.5  # Width of the person in meters (180 cm)
    # Convert alpha from degrees to radians
    alpha_degrees = 15  # Tilt angle in degrees
    alpha_radians = math.radians(alpha_degrees)

    f_pixels = 0.8  # Focal lens

    # Calculate distance
    distance = (H * f_pixels * frame_height) / (h_pixels * math.cos(alpha_radians))
    # distance = (W * f_pixels * frame_width) / (w_pixels * math.cos(alpha_radians))

    return distance


def drawTextOnFrame(frame, x, start_y, text_color_array):
    x = int(x)
    start_y = int(start_y)
    for i, (text, color) in enumerate(text_color_array):
        y = start_y + (i + 1) * 20  # Increment y by 20 pixels for each new line of text
        parts = text.split(": ")

        # Draw the first part in white
        cv2.putText(frame, parts[0] + ": ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw the second part in the specified color
        if len(parts) > 1:
            offset_x = x + cv2.getTextSize(parts[0] + ": ", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(frame, parts[1], (offset_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


# Create a class with fields: "id, image, bbox, last_time_seen, is_visible"
class MyTracker:
    def __init__(self, state, yolo_id, image, bbox, last_time_seen, is_visible):
        self.state = state
        self.yolo_id = yolo_id
        self.images = [image]
        self.bbox = bbox
        self.last_time_seen = last_time_seen
        self.is_visible = is_visible
        self.max_yolo_id = 0

        self.toast = None

        self.relock_max_new_yolo_id = 0
        self.relock_max_probability = 0
        self.relock_max_new_bbox = None

    def updateObject(self, frame, bbox):
        self.is_visible = True
        image = cutBboxFromFrame(frame, bbox)
        if len(self.images) > 20:
            self.images.pop(0)  # Remove the first element if length > 10
        self.images.append(image)  # Always add the new item to the en
        self.bbox = bbox
        self.last_time_seen = time.time()

    def getImage(self):
        if not self.images:
            return None
        return self.images[0]

    def showToast(self, text):
        self.toast = (text, time.time())

    def __repr__(self):
        return (f"TrackedObject(state={self.state}, bbox={self.bbox}, last_time_seen={self.last_time_seen}, "
                f"is_visible={self.is_visible})")


from enum import Enum


class TrackingState(Enum):
    INIT = "Initialization"
    LOCKED = "Locked"
    RE_LOCKING = "Re-locking"
    LOST = "Lost"

    def __str__(self):
        return self.value


def calculate_color_histogram_similarity(image1, image2):
    """
    Calculates the similarity between two images using color histograms.
    Resizes image2 so that its width matches image1 while keeping the height proportional.

    :param image1: The first image (numpy array).
    :param image2: The second image (numpy array).
    :return: A similarity score between 0 and 1, where 1 means identical.
    """
    if image1 is None or image2 is None:
        return 0
    # Resize image2 so that its width matches image1 while keeping height proportional
    if image1.shape[1] != image2.shape[1]:
        aspect_ratio = image2.shape[0] / image2.shape[1]  # Calculate the aspect ratio of image2
        new_width = image1.shape[1]  # Match the width of image1
        new_height = int(new_width * aspect_ratio)  # Calculate the proportional height
        image2 = cv2.resize(image2, (new_width, new_height))

    # Compute color histograms for each channel (BGR)
    hist1_b = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist1_g = cv2.calcHist([image1], [1], None, [256], [0, 256])
    hist1_r = cv2.calcHist([image1], [2], None, [256], [0, 256])

    hist2_b = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist2_g = cv2.calcHist([image2], [1], None, [256], [0, 256])
    hist2_r = cv2.calcHist([image2], [2], None, [256], [0, 256])

    # Normalize histograms
    hist1_b = cv2.normalize(hist1_b, hist1_b).flatten()
    hist1_g = cv2.normalize(hist1_g, hist1_g).flatten()
    hist1_r = cv2.normalize(hist1_r, hist1_r).flatten()

    hist2_b = cv2.normalize(hist2_b, hist2_b).flatten()
    hist2_g = cv2.normalize(hist2_g, hist2_g).flatten()
    hist2_r = cv2.normalize(hist2_r, hist2_r).flatten()

    # Compare histograms using correlation
    similarity_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
    similarity_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
    similarity_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)

    # Average the similarity scores for all channels
    similarity = (similarity_b + similarity_g + similarity_r) / 3

    return similarity


def calculateImageSimilarity(image1, image2):
    """
    Calculates the similarity between two images using Structural Similarity Index (SSIM).

    :param image1: The first image (numpy array).
    :param image2: The second image (numpy array).
    :return: A similarity score between 0 and 1, where 1 means identical.
    """
    # Convert images to grayscale for SSIM comparison
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same dimensions if needed
    if image1_gray.shape != image2_gray.shape:
        image2_gray = cv2.resize(image2_gray, (image1_gray.shape[1], image1_gray.shape[0]))

    # Calculate SSIM
    similarity, _ = ssim(image1_gray, image2_gray, full=True)

    return similarity


def create_and_save_side_by_side_image(image1, image2, similarity_score,
                                       output_dir="runs/debug/relocking/image_similarity", margin=10):
    """
    Combines two images side by side with a margin, resizes the combined image by 3x,
    adds the similarity score as text at the bottom, and saves the image to the specified folder
    with an incremented filename.

    :param image1: The first image (numpy array).
    :param image2: The second image (numpy array).
    :param similarity_score: The similarity score to display on the image.
    :param output_dir: The directory to save the combined image.
    :param margin: The margin space between the images (default is 10 pixels).
    """
    if image1 is None or image2 is None:
        return
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Resize the second image to match the first if needed
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Create a blank margin
    margin_shape = (image1.shape[0], margin, 3)
    margin_img = np.zeros(margin_shape, dtype=np.uint8)

    # Concatenate images side by side with margin
    combined_image = np.hstack((image1, margin_img, image2))

    # Resize the image 3x for better text rendering
    scale_factor = 6
    combined_image = cv2.resize(combined_image,
                                (combined_image.shape[1] * scale_factor, combined_image.shape[0] * scale_factor))

    # Add text with similarity score at the bottom
    text = f"Similarity Score: {similarity_score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Adjusted for larger image
    font_thickness = 3  # Adjusted for larger image
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (combined_image.shape[1] - text_size[0]) // 2
    text_y = combined_image.shape[0] - 20  # Slightly above the bottom edge

    # Put the text on the combined image
    cv2.putText(combined_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Determine the next filename based on the maximum number in the folder
    existing_files = [int(f.split('.')[0]) for f in os.listdir(output_dir) if f.split('.')[0].isdigit()]
    next_number = max(existing_files, default=0) + 1
    output_path = os.path.join(output_dir, f"{next_number}.jpg")

    # Save the combined image
    cv2.imwrite(output_path, combined_image)


def convertTensorBoxToInt(bbox):
    return int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())


def distance_between_bboxes(bbox1, bbox2):
    return math.sqrt(((bbox2[0] + bbox2[2]) / 2 - (bbox1[0] + bbox1[2]) / 2) ** 2 +
                     ((bbox2[1] + bbox2[3]) / 2 - (bbox1[1] + bbox1[3]) / 2) ** 2)


def cutBboxFromFrame(frame, bbox):
    return frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

def drawJoystick(frame, cx, cy, radius, joystick_x, joystick_x_max, joystick_y, joystick_y_max):
    bg_color = (0, 0, 0)
    active_color = (0, 0, 255)
    active_color_secondary = (0, 0, 200)
    cv2.circle(frame, (cx, cy), 5, bg_color, -1)
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), bg_color, 2)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), bg_color, 2)
    cv2.rectangle(frame, (cx - radius, cy - radius), (cx + radius, cy + radius), bg_color, 1)

    px_r = joystick_x
    if px_r > joystick_x_max:
        px_r = joystick_x_max
    elif px_r < -joystick_x_max:
        px_r = -joystick_x_max
    px_r = px_r / joystick_x_max

    py_r = joystick_y
    if py_r > joystick_y_max:
        py_r = joystick_y_max
    elif py_r < -joystick_y_max:
        py_r = -joystick_y_max
    py_r = py_r / joystick_y_max

    cv2.circle(frame, (int(cx + px_r * radius), int(cy + py_r * radius)), 10, active_color, -1)
    cv2.circle(frame, (int(cx + px_r * radius), int(cy + py_r * radius)), 10, active_color_secondary, 2)


    return frame



tracker = None

# Define the output video writer
output_path = "runs/output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' for H.264 or 'mp4v' for compatibility with .mp4
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Match the frame rate of the input video
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

out = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=True)

        # Get the boxes and track IDs
        boxes = []
        confidences = []
        track_ids = []
        if len(results) > 0:
            boxes = [convertTensorBoxToInt(box) for box in results[0].boxes.xyxy.cpu()]
            confidences = (results[0].boxes.conf * 100).int().tolist()

            if len(boxes) > 0 and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = frame.copy()

        # State handler

        # NO STATE
        if tracker is None:
            # Lock on random object
            tracker = MyTracker(TrackingState.INIT, None, None, None, None, True)
            continue

        # INIT
        if tracker.state == TrackingState.INIT:
            # Lock on the object that is closer to the center of the screen
            min_dist = None
            closest_to_center_yolo_id = None
            closest_to_center_bbox = None
            for yolo_id, bbox, confidence in zip(track_ids, boxes, confidences):
                x1, y1, x2, y2 = bbox
                x_dist = abs(x1 - (frame.shape[1] / 2))
                # # todo debug purposes
                # x_dist = x1
                if min_dist is None or x_dist < min_dist:
                    min_dist = x_dist
                    closest_to_center_yolo_id = yolo_id
                    closest_to_center_bbox = bbox
            # Saving
            if closest_to_center_yolo_id is not None:
                tracker.state = TrackingState.LOCKED
                tracker.yolo_id = closest_to_center_yolo_id
                tracker.updateObject(frame, closest_to_center_bbox)

        # LOCKED
        elif tracker.state == TrackingState.LOCKED:

            # Check if the target is found
            isTargetFound = False

            for yolo_id, bbox, confidence in zip(track_ids, boxes, confidences):
                if yolo_id != tracker.yolo_id:
                    continue
                # Current object
                x1, y1, x2, y2 = bbox
                tracker.updateObject(frame, bbox)
                isTargetFound = True

            # Update max_yolo_id parameter
            if len(track_ids) > 0:
                tracker.max_yolo_id = max(tracker.max_yolo_id, max(track_ids))

            if not isTargetFound:
                tracker.state = TrackingState.RE_LOCKING
                tracker.relock_max_new_yolo_id = 0
                tracker.relock_max_probability = 0
                tracker.relock_max_new_bbox = None
            else:
                color = (0, 255, 0)
                annotated_frame = cv2.rectangle(annotated_frame,
                                                (int(tracker.bbox[0]), int(tracker.bbox[1])),
                                                (int(tracker.bbox[2]), int(tracker.bbox[3])),
                                                color, 2)
                # Get the center points of the sides of the bounding box
                x1, y1, x2, y2 = map(int, tracker.bbox)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Draw arrows pointing from the sides to the center of the box
                padding = 15
                size = 15
                crosshair_radius = 3
                cv2.line(annotated_frame, (x1 - padding, center[1]), (x1 - padding - size, center[1] - size), color, 2)
                cv2.line(annotated_frame, (x1 - padding, center[1]), (x1 - padding - size, center[1] + size), color, 2)
                cv2.line(annotated_frame, (x2 + padding, center[1]), (x2 + padding + size, center[1] - size), color, 2)
                cv2.line(annotated_frame, (x2 + padding, center[1]), (x2 + padding + size, center[1] + size), color, 2)
                cv2.line(annotated_frame, (center[0] - crosshair_radius, center[1]),
                         (center[0] + crosshair_radius, center[1]), color, 2)
                cv2.line(annotated_frame, (center[0], center[1] - crosshair_radius),
                         (center[0], center[1] + crosshair_radius), color, 2)

                # Draw line from crosshair
                screen_center = int(frame.shape[1] / 2), center[1]
                cv2.circle(annotated_frame, screen_center, 3, (255, 0, 255), -1)
                cv2.line(annotated_frame, screen_center, center, (180, 0, 180), 2)

                distance_real = calculate_distance(frame.shape[0], tracker.bbox[3] - tracker.bbox[1])
                distance_feet = distance_real * 3.28084
                text_array = [
                    (f"Distance: {int(distance_real)}m / {int(distance_feet)}ft", (0, 240, 0)),
                ]
                annotated_frame = drawTextOnFrame(annotated_frame, int(tracker.bbox[2]) + 20, int(tracker.bbox[1]),
                                                  text_array)

                joystick_x = center[0] - screen_center[0]
                joystick_x_max = 50

                target_distance = 5
                joystick_y = target_distance - distance_real
                joystick_y_max = 5
                annotated_frame = drawJoystick(annotated_frame, frame.shape[1] - 120, frame.shape[0] - 120, 100, joystick_x, joystick_x_max, joystick_y, joystick_y_max)
                # annotated_frame = drawJoystick(annotated_frame, frame.shape[1] - 120, frame.shape[0] - 120, 100, joystick_x, joystick_x_max, joystick_y, joystick_y_max)

        # RE-LOCKING
        elif tracker.state == TrackingState.RE_LOCKING:
            # annotated_frame = results[0].plot()
            # Let yolo refind the same yolo_id object
            if time.time() - tracker.last_time_seen > 0.2:

                # Find every box possible and check if there are any closer
                similar_results = []
                probability_threshold = 0.5
                final_probability_threshold = 0.85

                for yolo_id, bbox, confidence in zip(track_ids, boxes, confidences):
                    # Check if this object is the same (recovered)
                    if yolo_id == tracker.yolo_id:
                        similar_results.append((1, yolo_id, bbox))  # Found same object
                        break
                    #
                    # # Check if this object is newly tracked
                    # if yolo_id < tracker.max_yolo_id:
                    #     continue

                    # Calculate probability that this object
                    # is the one that we were tracking
                    # probability = calculateProbability
                    distance_screen = distance_between_bboxes(bbox, tracker.bbox)
                    distance_normalized = 1 - (distance_screen * 1.0 / (frame.shape[1] / 2))

                    distance_real = calculate_distance(frame.shape[0], bbox[3] - bbox[1])
                    distance_real_tracker = calculate_distance(frame.shape[0], tracker.bbox[3] - tracker.bbox[1])
                    distance_real_normalized = 0
                    if distance_real > 0 and distance_real_tracker > 0:
                        # Normalize using a formula that returns 1 when distances are equal and approaches 0 as they differ
                        distance_real_normalized = 1 - abs(distance_real - distance_real_tracker) / max(distance_real,
                                                                                                        distance_real_tracker)
                    image_similarity = calculate_color_histogram_similarity(tracker.getImage(),
                                                                            cutBboxFromFrame(frame, bbox))
                    # create_and_save_side_by_side_image(tracker.getImage(), cutBboxFromFrame(frame, bbox), image_similarity)
                    # probability = distance_normalized * 0.2 + distance_real_normalized * 0.45 + image_similarity * 0.35
                    probability = distance_normalized * 0.3 + distance_real_normalized * 0.55 + image_similarity * 0.15
                    similar_results.append((probability, yolo_id, bbox))

                    print(
                        f"YOLO ID: {yolo_id}:\n\tDistance: {distance_screen:.2f}\n\tNormalized Distance: {distance_normalized:.2f}\n\tReal Distance: {distance_real:.2f}\n\tReal Distance Normalized: {distance_real_normalized:.2f}\n\tImage similaritye: {image_similarity:.2f}\n\tProbability: {probability:.2f}")

                    color = (0, 0, 255)
                    if yolo_id == tracker.relock_max_new_yolo_id:
                        color = (0, 255, 255)
                    annotated_frame = cv2.rectangle(annotated_frame,
                                                    (int(bbox[0]), int(bbox[1])),
                                                    (int(bbox[2]), int(bbox[3])),
                                                    color, 2)

                    text_array = [
                        (f"DIST SCREEN: {distance_normalized:.2f}", (200, 200, 200)),
                        (f"DIST REAL: {distance_real_normalized:.2f}", (200, 200, 200)),
                        (f"IMAGE SIM: {image_similarity:.2f}", (200, 200, 200)),
                        (f"PROB: {probability:.2f}",
                         (0, 255, 0) if probability >= probability_threshold else (0, 0, 255)),
                    ]
                    annotated_frame = drawTextOnFrame(annotated_frame, int(bbox[2]), int(bbox[1]) + 20,
                                                      text_array)

                # SORT SIMILAR_RESULTS tuples of (probability, yolo_id)
                similar_results.sort(reverse=True, key=lambda x: x[0])
                # Get the yolo_id with the highest probability
                if len(similar_results) > 0:
                    best_match = similar_results[0]
                    prob, yolo_id, bbox = best_match
                    if prob > tracker.relock_max_probability:
                        tracker.relock_max_probability = prob
                        tracker.relock_max_new_yolo_id = yolo_id
                        tracker.relock_max_new_bbox = bbox

                if tracker.relock_max_probability > final_probability_threshold:
                    tracker.state = TrackingState.LOCKED
                    tracker.yolo_id = tracker.relock_max_new_yolo_id
                    tracker.updateObject(frame, tracker.relock_max_new_bbox)
                    tracker.showToast(f"Found quick with prob = {tracker.relock_max_probability:.2f}")

                save_annotated_frame(annotated_frame)

                # WAIT AND EXIT
                if tracker.state == TrackingState.RE_LOCKING:
                    seconds_to_stop_finding = 5
                    seconds_track_most_reasonable_target_time = 3
                    t_diff = time.time() - tracker.last_time_seen
                    if t_diff >= seconds_track_most_reasonable_target_time:
                        if tracker.relock_max_probability > probability_threshold:
                            tracker.state = TrackingState.LOCKED
                            tracker.yolo_id = tracker.relock_max_new_yolo_id
                            tracker.updateObject(frame, tracker.relock_max_new_bbox)
                            tracker.showToast(f"Found after timeout with prob = {tracker.relock_max_probability:.2f}")
                    if t_diff >= seconds_to_stop_finding:
                        tracker.state = TrackingState.LOST
                        continue

        status_line = f"Status: {tracker.state.name}"
        status_color = (200, 200, 200)

        if tracker.state == TrackingState.LOCKED:
            status_color = (0, 255, 0)
        elif tracker.state == TrackingState.LOST:
            status_color = (0, 0, 255)
        elif tracker.state == TrackingState.RE_LOCKING:
            status_color = (0, 165, 255)

        text_array = [
            (status_line, status_color),  # Red for status
            (f"TRACKING YOLO ID: {tracker.yolo_id}", (0, 255, 0)),
        ]
        if tracker.state == TrackingState.RE_LOCKING:
            text_array.append((f"RE_LOCKING MAX PROB: {int(tracker.relock_max_probability * 100)}%", (0, 255, 0)))
        annotated_frame = drawTextOnFrame(annotated_frame, annotated_frame.shape[1] / 2 - 100, 100, text_array)

        if tracker.toast is not None:
            toast_message, toast_time = tracker.toast
            if time.time() - toast_time > 3:
                tracker.toast = None
            else:
                annotated_frame = drawTextOnFrame(annotated_frame, 100, annotated_frame.shape[0] - 100,
                                                  [(f"TOAST: {toast_message}", (0, 0, 255))])

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        out.write(annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

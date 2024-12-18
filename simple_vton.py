import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils

# Function to overlay transparent image
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """
    Overlay `img_overlay` onto `img` at position `pos` with transparency defined by `alpha_mask`.
    """
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha_mask_crop = alpha_mask[y1o:y2o, x1o:x2o] / 255.0
    alpha_inv = 1.0 - alpha_mask_crop

    # Perform alpha blending using NumPy operations for efficiency
    for c in range(0, 3):
        img_crop[:, :, c] = (alpha_inv * img_crop[:, :, c] +
                             alpha_mask_crop * img_overlay_crop[:, :, c])

    img[y1:y2, x1:x2] = img_crop

# Function to load and process clothing image
def load_clothing_image(image_path):
    clothing_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if clothing_img is None:
        raise ValueError(f"Clothing image not found at path: {image_path}")

    if clothing_img.shape[2] == 4:
        clothing_bgr = clothing_img[:, :, :3]
        alpha_mask = clothing_img[:, :, 3]
    else:
        clothing_bgr = clothing_img
        alpha_mask = np.ones(clothing_img.shape[:2], dtype=clothing_img.dtype) * 255

    return clothing_bgr, alpha_mask

# Function to calculate bounding boxes based on pose landmarks
def get_bounding_box(landmarks, image_width, image_height):
    """
    Calculate bounding boxes for shirt and pants based on pose landmarks.
    Top (shirt) is based on elbow-to-elbow width.
    Bottom (pants) is based on knee-to-knee width.
    Positioning is based on shoulders and torso.
    """
    # Extract relevant landmarks
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Convert normalized coordinates to pixel values
    def landmark_to_pixel(landmark):
        return int(landmark.x * image_width), int(landmark.y * image_height)

    le_x, le_y = landmark_to_pixel(left_elbow)
    re_x, re_y = landmark_to_pixel(right_elbow)
    lk_x, lk_y = landmark_to_pixel(left_knee)
    rk_x, rk_y = landmark_to_pixel(right_knee)
    ls_x, ls_y = landmark_to_pixel(left_shoulder)
    rs_x, rs_y = landmark_to_pixel(right_shoulder)
    lh_x, lh_y = landmark_to_pixel(left_hip)
    rh_x, rh_y = landmark_to_pixel(right_hip)

    # Calculate elbow-to-elbow width
    elbow_width = int(np.linalg.norm(np.array([le_x, le_y]) - np.array([re_x, re_y]))) * 1.4

    # Calculate knee-to-knee width
    knee_width = int(np.linalg.norm(np.array([lk_x, lk_y]) - np.array([rk_x, rk_y]))) * 1.3

    # Calculate torso height (shoulder to hip)
    torso_height = int(np.linalg.norm(np.array([ls_x, ls_y]) - np.array([lh_x, lh_y]))) * 1.15

    # Define padding factors
    shirt_padding = 0.1  # 10% padding
    pants_padding = 0.1  # 10% padding

    # Shirt bounding box based on torso and elbow widths
    shirt_top = min(ls_y, rs_y) - int(0.3 * torso_height)  # Slightly above shoulders
    shirt_bottom = max(lh_y, rh_y)  # At hips
    shirt_left = min(ls_x, rs_x) - int(shirt_padding * elbow_width)  # Slight padding
    shirt_right = max(ls_x, rs_x) + int(shirt_padding * elbow_width)

    # Pants bounding box based on torso and knee widths
    pants_top = shirt_bottom  # Start at hips
    pants_bottom = max(lk_y, rk_y) + int(0.1 * torso_height)  # Slightly below knees
    pants_left = min(lh_x, rh_x) - int(pants_padding * knee_width)  # Slight padding
    pants_right = max(lh_x, rh_x) + int(pants_padding * knee_width)

    return {
        'shirt': {
            'top': shirt_top,
            'bottom': shirt_bottom,
            'left': shirt_left,
            'right': shirt_right,
            'width': elbow_width
        },
        'pants': {
            'top': pants_top,
            'bottom': pants_bottom,
            'left': pants_left,
            'right': pants_right,
            'width': knee_width
        },
        'torso': {
            'top_shoulder_y': min(ls_y, rs_y),
            'bottom_hip_y': max(lh_y, rh_y)
        }
    }

# Function to calculate the top-left position for overlay based on shoulders and torso
def calculate_overlay_position_shirt(bounding_box, resized_clothing):
    """
    Calculate the top-left position where the shirt image should be placed based on shoulders and torso.
    """
    shirt_top = bounding_box['top']
    shirt_left = bounding_box['left'] + (bounding_box['right'] - bounding_box['left']) // 2 - resized_clothing.shape[1] // 2
    return shirt_left, shirt_top

# Function to overlay shirt image
def process_and_overlay_shirt(person_img, landmarks, bounding_boxes):
    # Load and process shirt image
    shirt_image_path = '/Users/smadu/Desktop/Hackathon/Simple_VTON/clothing_img/top1.jpg'
    shirt_bgr, shirt_alpha_mask = load_clothing_image(shirt_image_path)

    # Fit shirt to elbow-to-elbow width with padding (e.g., 5%)
    shirt_resized_bgr, shirt_resized_alpha = fit_clothing_to_limb_width(
        shirt_bgr, shirt_alpha_mask, bounding_boxes['shirt']['width'], padding=0.05)

    # Calculate shirt position based on shoulders and torso
    shirt_x_pos, shirt_y_pos = calculate_overlay_position_shirt(bounding_boxes['shirt'], shirt_resized_bgr)

    # Overlay shirt on person image
    overlay_image_alpha(person_img, shirt_resized_bgr, (shirt_x_pos, shirt_y_pos), shirt_resized_alpha)

# Function to resize clothing image to fit within bounding box based on limb width
def fit_clothing_to_limb_width(clothing_bgr, clothing_alpha, limb_width, padding=0.0):
    """
    Resize clothing image to fit based on limb width while maintaining aspect ratio.
    Padding can be added to slightly reduce the size to prevent overflow.
    """
    # Desired width based on limb measurement
    desired_width = int(limb_width * (1 + padding))  # Apply padding as a multiplier

    clothing_height, clothing_width = clothing_bgr.shape[:2]
    clothing_aspect_ratio = clothing_width / clothing_height

    # Determine new size while maintaining aspect ratio
    new_width = desired_width
    new_height = int(new_width / clothing_aspect_ratio)

    # Resize clothing image and alpha mask
    resized_bgr = cv2.resize(clothing_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_alpha = cv2.resize(clothing_alpha, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_bgr, resized_alpha

# Function to calculate the waist width of the person
def get_person_waist_width(landmarks, image_width, image_height):
    """
    Calculate the waist width of the person based on the pose landmarks.
    Uses the left and right hip landmarks.
    """
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Convert normalized coordinates to pixel values
    lh_x = int(left_hip.x * image_width)
    rh_x = int(right_hip.x * image_width)
    lh_y = int(left_hip.y * image_height)
    rh_y = int(right_hip.y * image_height)

    waist_width = abs(rh_x - lh_x)
    waist_y = int((lh_y + rh_y) / 2)

    return waist_width, lh_x, rh_x, waist_y

# Updated Function to calculate the top width of the pants image
def get_top_width_of_pants(alpha_mask):
    """
    Calculate the width of the pants at the top (waist) based on the alpha mask.
    Returns the pixel width and the leftmost and rightmost x-coordinates.
    Searches the top 20% of the image for the widest non-transparent row.
    """
    max_width = 0
    top_y = 0
    left_x = 0
    right_x = 0
    search_range = int(0.2 * alpha_mask.shape[0])  # Top 20% of the image

    for y in range(search_range):
        row = alpha_mask[y, :]
        if np.any(row > 0):
            x_indices = np.where(row > 0)[0]
            current_width = x_indices[-1] - x_indices[0]
            if current_width > max_width:
                max_width = current_width
                left_x = x_indices[0]
                right_x = x_indices[-1]
                top_y = y

    if max_width > 0:
        print(f"Found pants top width: {max_width} at row: {top_y}")
        return max_width, left_x, right_x, top_y
    else:
        print("No valid top width found in pants alpha mask.")
        return None, None, None, None

# Function to calculate the overlay position for pants
def calculate_overlay_position_pants(person_left_hip_x, person_right_hip_x, waist_y, resized_pants_width, pants_top_y, y_offset=0):
    """
    Calculate the top-left position where the pants image should be placed based on hips.
    The y_offset allows fine-tuning the vertical placement of the pants.
    """
    pants_x = person_left_hip_x + (person_right_hip_x - person_left_hip_x) // 2 - resized_pants_width // 2
    pants_y = waist_y - pants_top_y + y_offset  # Adjust for the top of the pants image and apply y_offset
    return pants_x, pants_y

# Updated Function to process and overlay pants image with error handling and y_offset
def process_and_overlay_pants(person_img, landmarks, image_width, image_height, y_offset=0):
    try:
        # Load and process pants image
        pants_image_path = '/Users/smadu/Desktop/Hackathon/Simple_VTON/clothing_img/shorts.png'
        pants_bgr, pants_alpha_mask = load_clothing_image(pants_image_path)

        # Get the top width of the pants image
        pants_top_width, pants_left_x, pants_right_x, pants_top_y = get_top_width_of_pants(pants_alpha_mask)
        if pants_top_width is None or pants_top_width == 0:
            raise ValueError("Invalid pants_top_width: cannot be zero or None.")

        # Set a minimum width to prevent division by zero
        MIN_WIDTH = 10  # Adjust as necessary
        pants_top_width = max(pants_top_width, MIN_WIDTH)
        if not np.isfinite(pants_top_width):
            raise ValueError("Pants top width is not finite.")

        # Get the waist width of the person
        waist_width, person_left_hip_x, person_right_hip_x, waist_y = get_person_waist_width(landmarks, image_width, image_height)
        print(f"Person waist width: {waist_width}")
        print(f"Person waist y-coordinate: {waist_y}")

        # Compute scaling factor
        scaling_factor = waist_width / pants_top_width
        print(f"Scaling factor: {scaling_factor}")

        # Handle infinite or excessively large scaling factors
        if not np.isfinite(scaling_factor):
            raise ValueError("Scaling factor is not finite.")

        # Resize pants image
        new_width = int(pants_bgr.shape[1] * scaling_factor)
        new_height = int(pants_bgr.shape[0] * scaling_factor)
        resized_pants_bgr = cv2.resize(pants_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_pants_alpha = cv2.resize(pants_alpha_mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Resized pants dimensions: {new_width}x{new_height}")

        # Calculate overlay position with y_offset
        pants_x_pos, pants_y_pos = calculate_overlay_position_pants(
            person_left_hip_x, person_right_hip_x, waist_y, new_width, pants_top_y, y_offset=y_offset)
        print(f"Pants overlay position: ({pants_x_pos}, {pants_y_pos})")

        # Overlay pants on person image
        overlay_image_alpha(person_img, resized_pants_bgr, (pants_x_pos, pants_y_pos), resized_pants_alpha)

        # Optional: Visualize the placement
        # Draw a circle at waist_y to verify alignment
        cv2.circle(person_img, (int((person_left_hip_x + person_right_hip_x) / 2), waist_y), 5, (0, 0, 255), -1)
        print("Drew waist_y marker on the image.")

    except Exception as e:
        print(f"Error in process_and_overlay_pants: {e}")

# Load mannequin image
person_image_path = '/Users/smadu/Desktop/Hackathon/Simple_VTON/person_img/manikin_black.jpg'
person_img = cv2.imread(person_image_path)
if person_img is None:
    raise ValueError(f"Mannequin image not found at path: {person_image_path}")

image_height, image_width = person_img.shape[:2]

# Perform pose detection
results = pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))

if not results.pose_landmarks:
    raise ValueError("No pose landmarks detected.")

# Visualize pose landmarks on the image (optional)
annotated_image = person_img.copy()
mp_drawing.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
cv2.imwrite('pose_landmarks.png', annotated_image)
print("Pose landmarks image saved to pose_landmarks.png")

# Extract key landmarks
landmarks = results.pose_landmarks.landmark

# Calculate bounding boxes
bounding_boxes = get_bounding_box(landmarks, image_width, image_height)

# Define y_offset for pants positioning
# Adjust this value based on your specific clothing image
# Positive values move pants downward, negative values move them upward
y_offset = 10  # Example value; you may need to adjust this

# Process and overlay pants image
# process_and_overlay_pants(annotated_image, landmarks, image_width, image_height, y_offset=y_offset)

# Process and overlay shirt image
process_and_overlay_shirt(annotated_image, landmarks, bounding_boxes)

# Save the final result
output_path = 'virtual_try_on_result_with_landmarks.png'
cv2.imwrite(output_path, annotated_image)
print(f"Virtual try-on result saved to {output_path}")

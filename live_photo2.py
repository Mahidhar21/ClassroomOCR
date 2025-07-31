
import streamlit as st
import cv2
import numpy as np
import os
from tqdm import tqdm
from ultralytics import YOLO
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Blackboard Content Extractor")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# if uploaded_file is not None:
#     # Save the uploaded video to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         video_path = tmp_file.name
# --- Live Photo Video Buffer ---
class VideoBuffer(VideoTransformerBase):
    def __init__(self):
        self.frames = []
        self.fps = 30  # Assume 30fps
        self.max_seconds = 5  # Keep only last 5 seconds
        self.max_frames = self.fps * self.max_seconds
        self.capture_triggered = False
        self.captured_frames = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if len(self.frames) >= self.max_frames:
            self.frames.pop(0)
        self.frames.append(img.copy())
        if self.capture_triggered:
            # Only keep the last max_frames (i.e., last 5 seconds)
            self.captured_frames = self.frames[-self.max_frames:].copy()
            self.capture_triggered = False
        return img


# --- Streamlit UI ---
# st.title("Live Photo Capture Demo (Buffering Moments Before Click)")

st.write("Try to hold your phone still for better results. Point your camera at the board and press the shutter button just after the person obstructing the board moves. Our powerful eraser shall remove the obstructions and give you the complete picture of board while retaining the same text!")

if 'camera_mode' not in st.session_state:
    st.session_state['camera_mode'] = False
if 'show_live_photo' not in st.session_state:
    st.session_state['show_live_photo'] = False
if 'captured_frames' not in st.session_state:
    st.session_state['captured_frames'] = None
if 'live_photo_video_path' not in st.session_state:
    st.session_state['live_photo_video_path'] = None


# Only show the Start Camera button if camera is not running
if not st.session_state['camera_mode']:
    if st.button('Classroom OCR'):
        st.session_state['camera_mode'] = True

# Only show the camera UI if camera_mode is True
if st.session_state['camera_mode']:
    video_buffer = VideoBuffer()
    webrtc_ctx = webrtc_streamer(
        key="livephoto-demo",
        video_transformer_factory=lambda: video_buffer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Capture Live Photo"):
                webrtc_ctx.video_transformer.capture_triggered = True
                time.sleep(0.2)  # Let the buffer update
                frames = webrtc_ctx.video_transformer.captured_frames
                # Stop the camera and clear buffer after capture
                st.session_state['camera_mode'] = False
                if frames:
                    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    height, width, _ = frames[0].shape
                    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                    for f in frames:
                        out.write(f)
                    out.release()
                    st.session_state['captured_frames'] = frames
                    st.session_state['live_photo_video_path'] = temp_video.name
                    st.session_state['show_live_photo'] = True
                    st.success("Live photo captured!")
                else:
                    st.error("No frames captured. Please try again.")
                st.rerun()
        with col2:
            if st.button('Close Camera'):
                st.session_state['camera_mode'] = False
                st.session_state['captured_frames'] = None
                st.session_state['live_photo_video_path'] = None
                st.session_state['show_live_photo'] = False
                st.rerun()

# Only display the image after capture, not before

# Use the saved video path from the buffer for further processing
live_photo_video_path = st.session_state.get('live_photo_video_path')
if st.session_state.get('show_live_photo') and st.session_state.get('captured_frames') and live_photo_video_path:
    frames = st.session_state['captured_frames']
    st.image(frames[-1], caption="Live Photo (Last Frame)", use_container_width=True)
    st.info(f"Live photo video saved at: {live_photo_video_path}")
    # Reset flag so image is not shown again until next capture
    st.session_state['show_live_photo'] = False

    # Video processing pipeline using the saved video path
    try:
        model = YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'best.pt' is in the same directory or use a different model path.")
        st.stop()

    output_frame_dir = "/tmp/frames"
    os.makedirs(output_frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(live_photo_video_path)
    frames = []
    idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Finding obstructions")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        if "board" in labels and "person" not in labels:
            frames.append(frame)
        idx += 1
        progress_bar.progress(min(idx / max(total_frames, 1), 1.0), text="Finding obstructions")
    cap.release()

    if len(frames) == 0:
        st.error("No clean frames found with board only. Please check your video or model.")
    else:
        base = frames[0]
        aligned_stack = []

        def align_frames(reference, target):
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(500)
            kp1, des1 = orb.detectAndCompute(reference_gray, None)
            kp2, des2 = orb.detectAndCompute(target_gray, None)
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return None
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if des1 is not None and des2 is not None:
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
            else:
                return None
            if len(matches) < 10:
                st.warning(f"Skipping alignment due to insufficient matches ({len(matches)}).")
                return None
            src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return None
            return cv2.warpPerspective(target, H, (reference.shape[1], reference.shape[0]))

        for i in range(len(frames)):
            if i == 0:
                aligned_stack.append(base)
                continue
            aligned = align_frames(base, frames[i])
            if aligned is not None:
                aligned_stack.append(aligned)
            progress_bar.progress(min((i + 1) / max(len(frames), 1), 1.0), text="Erasing the person")

        if len(aligned_stack) == 0:
            st.error("No frames were successfully aligned. Cannot proceed.")
        else:
            try:
                person_model = YOLO('yolov8n.pt')
            except Exception as e:
                st.error(f"Error loading person detection model: {e}. Make sure 'yolov8n.pt' is available.")
                st.stop()
            person_masks = []
            for i, frame in enumerate(aligned_stack):
                results = person_model(frame, verbose=False)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for r in results[0].boxes:
                    if person_model.names[int(r.cls)] == 'person':
                        x1, y1, x2, y2 = [int(i) for i in r.xyxy[0]]
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                person_masks.append(mask)
                progress_bar.progress(min((i + 1) / max(len(aligned_stack), 1), 1.0), text="Erasing the person")

            if len(person_masks) != len(aligned_stack):
                st.error("Mismatch between number of person masks and aligned frames.")
            else:
                fused_board_selective = np.zeros_like(aligned_stack[0], dtype=np.float32)
                count_matrix = np.zeros(aligned_stack[0].shape[:2], dtype=np.float32)
                for i, frame in enumerate(aligned_stack):
                    inverted_mask = cv2.bitwise_not(person_masks[i])
                    masked_frame = cv2.bitwise_and(frame, frame, mask=inverted_mask)
                    fused_board_selective += masked_frame.astype(np.float32)
                    count_matrix += (inverted_mask > 0).astype(np.float32)
                    progress_bar.progress(min((i + 1) / max(len(aligned_stack), 1), 1.0), text="Erasing the person")
                count_matrix_expanded = np.expand_dims(count_matrix, axis=-1)
                count_matrix_expanded[count_matrix_expanded == 0] = 1
                fused_board_selective /= count_matrix_expanded
                fused_board_selective = fused_board_selective.astype(np.uint8)
                st.sidebar.header("Post-processing Options")
                denoising_h = st.sidebar.slider("Denoising Strength (h)", 0, 50, 10)
                sharpening_strength = st.sidebar.slider("Sharpening Strength", 0.0, 5.0, 1.0)
                denoised_image = cv2.fastNlMeansDenoisingColored(fused_board_selective, None, denoising_h, denoising_h, 7, 21)
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=np.float32) * sharpening_strength
                kernel[1, 1] += (1 - sharpening_strength)
                sharpened_image = cv2.filter2D(denoised_image, -1, kernel)
                final_processed_image = sharpened_image
                progress_bar.empty()
                st.write("Person erased succesfully.")
                st.image(final_processed_image, channels="BGR")
    os.unlink(live_photo_video_path)
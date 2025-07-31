import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
import time
import tempfile
from collections import deque

# --- Constants ---
FRAME_BUFFER_SIZE = 90  # 3 seconds at 30 fps
PRE_CAPTURE_FRAMES = 45
POST_CAPTURE_FRAMES = 45

# --- Video Transformer ---
class VideoBuffer(VideoTransformerBase):
    def __init__(self):
        self.buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.capture_triggered = False
        self.capture_complete = False
        self.captured_frames = []
        self.lock = threading.Lock()
        self.post_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.buffer.append(img.copy())
            if self.capture_triggered and not self.capture_complete:
                if self.post_count == 0:
                    self.captured_frames = list(self.buffer)[-PRE_CAPTURE_FRAMES:]
                if self.post_count < POST_CAPTURE_FRAMES:
                    self.captured_frames.append(img.copy())
                    self.post_count += 1
                if self.post_count >= POST_CAPTURE_FRAMES:
                    self.capture_complete = True
                    self.capture_triggered = False
                    self.post_count = 0
        return img

# --- App Start ---
st.title("📸 Live Photo Streamlit App")

# Initialize session state
if 'camera_mode' not in st.session_state:
    st.session_state['camera_mode'] = False
if 'show_live_photo' not in st.session_state:
    st.session_state['show_live_photo'] = False
if 'video_buffer' not in st.session_state:
    st.session_state['video_buffer'] = None

# --- Start Camera ---
if not st.session_state['camera_mode'] and not st.session_state['show_live_photo']:
    if st.button("Start Camera"):
        st.session_state['camera_mode'] = True
        st.session_state['video_buffer'] = VideoBuffer()
        st.rerun()

# --- Camera Mode ---
if st.session_state['camera_mode']:
    st.subheader("Live Camera")

    video_buffer = st.session_state['video_buffer']

    webrtc_ctx = webrtc_streamer(
        key="livephoto-demo",
        video_transformer_factory=lambda: video_buffer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    if webrtc_ctx.video_transformer and webrtc_ctx.state.playing:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Capture Live Photo"):
                video_buffer.capture_triggered = True
                time.sleep(0.5)  # Let it buffer
                captured = video_buffer.captured_frames.copy()
                if captured:
                    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    height, width, _ = captured[0].shape
                    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                    for f in captured:
                        out.write(f)
                    out.release()

                    st.session_state['captured_frames'] = captured
                    st.session_state['live_photo_video_path'] = temp_video.name
                    st.session_state['show_live_photo'] = True
                    st.session_state['camera_mode'] = False
                    st.success("Live photo captured!")
                    st.rerun()
                else:
                    st.error("No frames captured. Please try again.")
        with col2:
            if st.button('Close Camera'):
                st.session_state['camera_mode'] = False
                st.session_state['video_buffer'] = None
                st.rerun()

# --- Display Live Photo ---
if st.session_state.get('show_live_photo'):
    st.subheader("📷 Your Live Photo")
    video_path = st.session_state.get('live_photo_video_path')
    if video_path:
        st.video(video_path)

    if st.button("Retake"):
        st.session_state['camera_mode'] = True
        st.session_state['show_live_photo'] = False
        st.session_state['video_buffer'] = VideoBuffer()
        st.rerun()

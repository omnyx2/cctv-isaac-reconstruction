"""Convert AI Hub CCTV images to video."""
import cv2
import numpy as np
from pathlib import Path

img_dir = Path(r'C:/Users/omnyx/CCTV_Project/data/교통안전(Bbox)/원천데이터/삼정고가교/BC1000801')
images = sorted(img_dir.glob('*.jpg'))
print(f'Found {len(images)} images')

# Read first image using binary mode (for Korean path support)
with open(images[0], 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8)
    first = cv2.imdecode(data, cv2.IMREAD_COLOR)

h, w = first.shape[:2]
print(f'Image size: {w}x{h}')

# Create video
out_path = r'C:/Users/omnyx/CCTV_Project/data/videos/aihub_cctv.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, 10, (w, h))  # 10 FPS

for i, img_path in enumerate(images[:300]):  # First 300 frames
    with open(img_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is not None:
        out.write(frame)
    if i % 50 == 0:
        print(f'Processing {i}/{min(300, len(images))}')

out.release()
print(f'Created: {out_path}')

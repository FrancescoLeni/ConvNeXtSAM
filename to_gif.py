import imageio
import os

# === Configuration ===

video_path = fr"C:\Users\franc\Videos\{n}.mp4"
gif_path = fr'repo_data/{n}.gif'
frame_step = 1      # Use every N-th frame (e.g., skip every other frame)
resize = None # Resize frames to width x height (optional)

# === Convert video to GIF ===
reader = imageio.get_reader(video_path)
fps = reader.get_meta_data()['fps'] * 1.5

frames = []
for i, frame in enumerate(reader):
    if i % frame_step == 0:
        if resize:
            from PIL import Image
            frame = Image.fromarray(frame).resize(resize, Image.ANTIALIAS)
            frame = frame.convert("RGB")  # Ensure no alpha channel
            frames.append(frame)
        else:
            frames.append(frame)

reader.close()

# Save GIF
imageio.mimsave(gif_path, frames, fps=fps // frame_step, loop=0)

print(f"✅ Saved GIF to: {os.path.abspath(gif_path)}")

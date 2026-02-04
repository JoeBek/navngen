import ffmpeg
import numpy as np
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
from typing import Union
import cv2
from tqdm import tqdm




class VideoDecoder:
    def __init__(self, video_path, width=None, height=None):
        self.video_path = video_path
        self.probe = ffmpeg.probe(video_path)
        self.video_info = next(s for s in self.probe['streams'] if s['codec_type'] == 'video')
        
        self.original_width = int(self.video_info['width'])
        self.original_height = int(self.video_info['height'])
        self.fps = eval(self.video_info['r_frame_rate'])
        
        self.width = width or self.original_width
        self.height = height or self.original_height
        
    def __iter__(self):
        """Iterator for all frames."""
        process = (
            ffmpeg
            .input(self.video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.width}x{self.height}')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        frame_size = self.width * self.height * 3
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            yield frame
        
        process.wait()
    
    def get_frame_at_time(self, timestamp):
        """Get a single frame at specific timestamp (in seconds)."""
        out, _ = (
            ffmpeg
            .input(self.video_path, ss=timestamp)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1, s=f'{self.width}x{self.height}')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        frame = np.frombuffer(out, np.uint8).reshape([self.height, self.width, 3])
        return frame
    
    def get_frames_batch(self, start_time, end_time):
        """Get frames between start and end time."""
        duration = end_time - start_time
        process = (
            ffmpeg
            .input(self.video_path, ss=start_time, t=duration)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.width}x{self.height}')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        frames = []
        frame_size = self.width * self.height * 3
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            frames.append(frame)
        
        process.wait()
        return frames


def fig_to_numpy(fig, close=True):
    """Return RGB uint8 (H, W, 3) from a Matplotlib figure."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()
    if close:
        plt.close(fig)
    return rgb


def apply_sp(frame: np.ndarray, debug=False) -> Union[np.ndarray, None]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
        
    tensor = numpy_image_to_torch(frame)
    feats0 = extractor.extract(tensor.to(device))
    kpts0 = feats0["keypoints"]

    viz2d.plot_images([frame])
    viz2d.plot_keypoints(kpts0, ps=6)

    if debug:
        plt.show()
        return
    # get plot as numpy
   
    fig = plt.gcf()
    kp_frame = fig_to_numpy(fig)
    return kp_frame


def apply_sp_frames(frames):
    
    new_frames = []
    for frame in tqdm(frames):
        new_frame = apply_sp(frame)
        new_frames.append(new_frame)

    return new_frames

        
def encode_video(
    frames: list[np.ndarray],
    output_path: Path | str,
    fps: int = 60,
    crf: int = 16,
    preset: str = "medium",
):
    """
    Encode a list of RGB uint8 frames (H, W, 3) into an MP4 (H.264).
    """
    if not frames:
        raise ValueError("frames list is empty")

    # Ensure consistent size and dtype
    h, w = frames[0].shape[:2]
    # libx264 / yuv420p requires even dimensions. Pad if odd to avoid
    # "width not divisible by 2" errors from ffmpeg.
    target_w = w + (w % 2)
    target_h = h + (h % 2)
    fixed = []
    for f in frames:
        if f.dtype != np.uint8:
            f = f.astype(np.uint8)
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        # pad to even dims if required
        if (target_w != w) or (target_h != h):
            pad_w = target_w - w
            pad_h = target_h - h
            # pad right and bottom with black pixels
            f = cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if f.ndim == 2:
            f = np.repeat(f[..., None], 3, axis=2)
        if f.shape[2] != 3:
            raise ValueError("Frames must have 3 channels (RGB)")
        fixed.append(f)

    out = str(output_path)
    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{target_w}x{target_h}", r=fps)
        .output(
            out,
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=crf,
            preset=preset,
            r=fps,
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    # Write frames to ffmpeg stdin, but capture stderr so we can report
    # ffmpeg errors (BrokenPipeError typically means ffmpeg exited early).
    stderr_bytes = b""
    try:
        for f in fixed:
            try:
                process.stdin.write(f.tobytes())
            except BrokenPipeError:
                # ffmpeg closed the pipe; read stderr for diagnostics
                try:
                    stderr_bytes = process.stderr.read() or b""
                except Exception:
                    stderr_bytes = b""
                msg = stderr_bytes.decode("utf-8", errors="replace")
                raise RuntimeError(
                    "ffmpeg closed the pipe while writing frames. "
                    "ffmpeg stderr:\n\n" + msg
                )
    finally:
        # always close stdin and wait for process
        try:
            process.stdin.close()
        except Exception:
            pass
        try:
            # try to read remaining stderr
            if not stderr_bytes:
                stderr_bytes = process.stderr.read() or b""
        except Exception:
            pass
        ret = process.wait()

    if ret != 0:
        msg = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else f"ffmpeg exited with code {ret}"
        raise RuntimeError(f"ffmpeg failed: {msg}")


if __name__ == "__main__":

    path = Path("/home/joe/vt/research/glue/assets/vt_data/video_baselines/durham_cropped.mp4")
    
    decoder = VideoDecoder(path)
    
    frames = list(decoder)    
    new_frames = apply_sp_frames(frames)
    
    out_path = Path("durham_keypoint.mp4")

    encode_video(new_frames, out_path)

    

    
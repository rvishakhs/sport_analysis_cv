from Basketball_analysis.drawers.utils import draw_traiangle


class BallTrackDrawer:
    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            output_frame = frame.copy()
            ball_dict = tracks[frame_num]

            for _, track in ball_dict.items():
                bbox = track['bbox']
                if bbox is None:
                    continue

                output_frame = draw_traiangle(frame, bbox, self.ball_pointer_color)

            output_video_frames.append(output_frame)

        return output_video_frames


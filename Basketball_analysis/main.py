from pprint import pprint

from ultralytics import YOLO
from multiprocessing import freeze_support

from Basketball_analysis.drawers import PlayerTracksDrawer, BallTrackDrawer
from Basketball_analysis.team_assign import TeamAssigner
from Basketball_analysis.trackers.ball_tracker import BallTracker
from utils import read_video, save_video
from trackers import PlayerTracker

def main():
# Read video
    video_frames = read_video('input_video/video_1.mp4')

# Initialize the trackers
    player_tracker = PlayerTracker("models/player_detection_model.pt")
    ball_tracker = BallTracker("models/ball_detection_model.pt")

    player_tracks = player_tracker.get_object_track(video_frames,
                                                    read_from_stub=True,
                                                    stub_path="stubs/player_track_stubs.pkl")

    ball_tracks = ball_tracker.get_object_track(video_frames,
                                                read_from_stub=True,
                                                stub_path="stubs/ball_track_stubs.pkl")

    team_assigner = TeamAssigner()
    playerteams = team_assigner.get_player_teams_across_frames(video_frames,
                                                player_tracks,
                                                read_from_stub=True,
                                                stub_path="stubs/Team_assigner_stubs.pkl"
                                                )

    # Remove Wrong Tracking
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    # Interpolations for better ball tracking use only if needed
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)



# Draw tracks
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTrackDrawer()
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, playerteams)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)


# Save Video
    save_video(output_video_frames, 'output_video/video.mp4')


if __name__ == '__main__':
    freeze_support()
    main()

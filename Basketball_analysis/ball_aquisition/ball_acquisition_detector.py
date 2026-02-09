from easyocr.utils import consecutive

from Basketball_analysis.utils import measure_distance


class BallAcquisitionDetector:
    def __init__(self):
        self.possession_threshold = 50
        self.min_frames = 11
        self.containment_threshold = 0.8

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        ball_center_x, ball_center_y =  ball_center[0], ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        output_points = []

        if y1 < ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if x1 < ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))


        output_points += [
            (x1, y1) # top left corner
            (x2, y1) # top right corner
            (x1, y2) # bottom left corner
            (x2, y2) # Bottom right corner
            (x1 + width //2, y1 ) # Top center
            (x1 + width //2, y2 ) # bottom center
            (x1, y1+height //2 ) # left Center
            (x2, y1+height //2 ) # right Center
        ]

        return output_points

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        keypoints = self.get_key_basketball_player_assignment_points(ball_center, player_bbox)

        return min(measure_distance(ball_center, keypoint) for keypoint in keypoints)

    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ball_area = (bx2 - bx1) * (by2 - by1)

        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

        containment_ratio = intersection_area / ball_area

        return containment_ratio

    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):

        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info['bbox']
            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, containment))
            else:
                regular_distance_players.append((player_id, containment))


        # First priority high_containment_players
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]

        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]

        return -1

    def detetct_ball_possession(self, player_tracks, ball_tracks):
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        consecutive_possession_count = {}

        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue




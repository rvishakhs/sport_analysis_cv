import cv2
import numpy as np

from Basketball_analysis.utils import get_center_of_bbox, get_bbox_width


def draw_elipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, y_center = map(int, get_center_of_bbox(bbox))
    width = int(get_bbox_width(bbox))

    cv2.ellipse(img = frame,
                center=(x_center, y2),
                axes=(int(width), int(0.35*width)),
                angle=0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
                )

    rect_width = 40
    rect_height = 20

    x1_rect = x_center-rect_width//2
    x2_rect = x_center + rect_width//2
    y1_rect = (y2 - rect_height//2)+15
    y2_rect = (y2 + rect_height//2)+15

    if track_id is not None:
        cv2.rectangle(frame,
                      (x1_rect, y1_rect),
                      (x2_rect, y2_rect),
                      color,
                      cv2.FILLED)

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(frame, str(track_id), (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return frame


def draw_traiangle(frame, bbox, ball_pointer_color):
    y = int(bbox[1])
    x, _ = map(int, get_center_of_bbox(bbox))

    trinagle_points = np.array([
        [x,y],
        [x-10, y-20],
        [x+10, y-20],
    ])

    cv2.drawContours(frame,[trinagle_points], 0, ball_pointer_color, cv2.FILLED)
    cv2.drawContours(frame,[trinagle_points], 0, ball_pointer_color, 2)

    return frame






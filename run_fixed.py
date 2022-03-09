import cv2

from config import CASCADES


def is_user_wants_quit():
    return cv2.waitKey(1) & 0xFF == ord("q")


def show_frame(frame):
    cv2.imshow("Video", frame)


def draw_sqare(frame, region, color):
    left, top, width, height = region
    cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)


def get_cascades():
    cascades = [
        (cv2.CascadeClassifier(cascade["path"]), cascade["color"])
        for _, cascade in CASCADES.items()
        if cascade["draw"]
    ]
    return cascades


if __name__ == "__main__":
    cascades = get_cascades()
    video_capture = cv2.VideoCapture(0)
    while True:
        if not video_capture.isOpened():
            print("Couldn't find your webcam... Sorry :c")
        _, webcam_frame = video_capture.read()
        gray_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)
        for cascade, color in cascades:
            detected_regions = cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30)
            )
            for region in detected_regions:
                draw_sqare(webcam_frame, region, color)

        show_frame(webcam_frame)

        if is_user_wants_quit():
            break
    video_capture.release()
    cv2.destroyAllWindows()

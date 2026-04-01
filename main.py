import argparse
import sys
import time

import cv2

from config import DEFAULT_SOURCE
from core.validator import NFCValidator
from utils.roi_utils import select_roi


def parse_args():
    parser = argparse.ArgumentParser(description="NFC Optical Validator Demo")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["led", "multi_led", "ocr"],
        help="Detection mode",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="Camera index or video file path",
    )
    return parser.parse_args()


def open_source(source_arg: str):
    if source_arg.isdigit():
        return cv2.VideoCapture(int(source_arg))
    return cv2.VideoCapture(source_arg)


def main():
    args = parse_args()

    cap = open_source(args.source)
    if not cap.isOpened():
        print("Error: could not open source")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: could not read first frame")
        cap.release()
        sys.exit(1)

    roi = select_roi(first_frame, window_name="Select ROI")
    if roi is None:
        print("ROI selection cancelled")
        cap.release()
        sys.exit(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    validator = NFCValidator(mode=args.mode, roi=roi, fps=fps)
    validator.start_recording(first_frame.shape[1], first_frame.shape[0])

    frame_index = 0
    fps_timer = time.time()
    fps_counter = 0
    measured_fps = fps if fps and fps > 1 else 20.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter += 1
        now = time.time()
        elapsed_for_fps = now - fps_timer
        if elapsed_for_fps >= 1.0:
            measured_fps = fps_counter / elapsed_for_fps
            fps_timer = now
            fps_counter = 0
            validator.fps = measured_fps

        result = validator.process_frame(frame, frame_index)
        annotated_frame = result["annotated_frame"]

        cv2.imshow("NFC Optical Validator", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if result["finalized"]:
            print("Final result:", result["decision"]["result"])
            print("Reason:", result["decision"]["reason"])
            if result["saved_paths"]:
                print("Saved:")
                for k, v in result["saved_paths"].items():
                    print(f"  {k}: {v}")

            cv2.imshow("NFC Optical Validator", annotated_frame)
            cv2.waitKey(1500)
            break

        frame_index += 1

    if validator.recorder.is_active():
        validator.recorder.stop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
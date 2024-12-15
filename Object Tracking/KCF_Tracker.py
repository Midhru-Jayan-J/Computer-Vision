import cv2

tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture('Object Tracking\\Data - Videos\\race.mp4')

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

ok, frame = video.read()
if not ok:
    print("Error: Cannot read the first frame from the video.")
    exit()
    
bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    
    if not ok:
        break

    ok, bbox = tracker.update(frame)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, 'Error', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()



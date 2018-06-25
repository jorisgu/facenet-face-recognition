import cv2
import sys
import time

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


class face_detector:
    def __init__(self,time_wait=1):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.faces = []
        self.last_update = 0
        self.time_wait = time_wait

    def detect_face(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def update_faces(self,frame):
        if time.time()-self.last_update>self.time_wait:
            print("Updating faces")
            self.faces = self.detect_face(frame)
            self.last_update = time.time()

class face_trackers:
    def __init__(self):
        self.trackers = []
            # Set up tracker.
            # Instead of MIL, you can also use

        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_types[1]

    def add_tracker(self,frame,bbox):
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if self.tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if self.tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if self.tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if self.tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if self.tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if self.tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
        selF.trackers.append({"tracker":tracker.init(frame, bbox)})

    def update_trackers(self,frame):
        for t in self.trackers:
            t['ok'], t['bbox'] = t['tracker'].update(frame)

    def already_tracked(self,bbox):
        for t in self.trackers:
            joris il faut rajouter le calcul des aires (voir dans maskrcnn)
            if compute_iou(bbox,t['bbox'])>0.8:
                return True
        return False


if __name__ == '__main__' :



    # Read video

        # cv2.namedWindow("preview")

    # print("Using",tracker_type)
    # video = cv2.VideoCapture("videos/chaplin.mp4")
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    # bbox = (287, 23, 86, 320)
    # bbox = (300,300,400,400)
        # x1,y1_____________
        # |                 |
        # |                 |
        # |_________________x2,y2


    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame)
    bbox = None
    # print(bbox)tracker

    # Initialize tracker with first frame and bounding box
    # ok = tracker.init(frame, bbox)

    fd = face_detector()
    ft = face_trackers()
    while True:

        # Start timer
        timer = cv2.getTickCount()


        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break


        ft.update_trackers(frame)


        fd.update_faces(frame)


        for face in faces:
            if face not already tracked:
                add tracker
            else:
                pass
        if bbox is None:
            if len(fd.faces)>0:
                print("faces here:",fd.faces)

        # Update tracker
        # ok, bbox = tracker.update(frame)
        # print(bbox)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        # if ok:
            # Tracking success
            # p1 = (int(bbox[0]), int(bbox[1]))
            # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            # cv2.rectangle(frame, p1, p2, (255,255,0), 2, 1)
        # else :
            # Tracking failure
            # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        # cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

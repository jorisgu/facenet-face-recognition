import cv2
import sys
import time

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

class face_brain:
    def __init__(self,time_wait=5):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.faces = []
        self.last_update = 0
        self.time_wait = time_wait
        self.used = False


        self.trackers = []
        self.tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = self.tracker_types[1]

    def detect_face(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def update_faces(self,frame):
        if time.time()-self.last_update>self.time_wait:
            print("Updating faces")
            self.faces = self.detect_face(frame)
            self.last_update = time.time()
            self.used = False

    def get_faces(self):
        if self.used:
            return []
        else:
            self.used = True
            return self.faces

    def add_tracker(self,frame,bbox):
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(self.tracker_type)
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
        # print(bbox)
        tracker.init(frame, tuple(bbox))
        self.trackers.append({"tracker":tracker,"bbox":bbox})

    def update_trackers(self,frame):
        for t in self.trackers:
            t['ok'], t['bbox'] = t['tracker'].update(frame)
        self.trackers = [t for t in self.trackers if t['ok']]

    def print_status(self,text="",debug=False):
        if debug:
            print("Nombre de trackers :", len(self.trackers))
            print("Nombre de faces    :", len(self.faces))
            print(40*"-")

    def already_tracked(self,bbox):
        for t in self.trackers:
            iou=bb_intersection_over_union(bbox,t['bbox'])
            print(iou)
            if iou>0.5:
                return True
        return False

    def update_frame(self,frame):
        self.print_status("0 : update_trackers")
        self.update_trackers(frame)
        self.print_status("1 : updates_faces")
        self.update_faces(frame)
        self.print_status("2 : already tracked ?")
        for face in self.get_faces():
            print("About a face :",face)
            if not self.already_tracked(face):
                print("new tracking")
                self.add_tracker(frame,face)
            else:
                print("already tracked")



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

    fb = face_brain()
    while True:

        # Start timer
        timer = cv2.getTickCount()


        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break


        fb.update_frame(frame)


        # Update tracker
        # ok, bbox = tracker.update(frame)
        # print(bbox)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);


        for id_t,t in enumerate(fb.trackers):

            # print("drawing",t)
            bbox = t['bbox']
        # Draw bounding box
        # if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,255,0), 2, 1)
            p1_text = (int(bbox[0])+5, int(bbox[1]))
            cv2.putText(frame, str(id_t), p1_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,255),2)
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

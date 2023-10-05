
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

DATA_SET = "PETS"

# Paremeters
video_index =0
if DATA_SET == "AOD":
    # Abandoned object detection parameters

    videos = ["AOD_Datasets/Set_1/video" + str(i) + ".avi" for i in range(1,12)]
    start_frames = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    buffer_scales = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    short_buffer_sizes = [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250]
    long_buffer_sizes= [350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350]
    sample_rates = [5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 8, 5]        # Time compression - which changes the effect of the buffer lengths
    obstacle_thresholds = [0.0025, 0.00005, 0.0001, 0.00005, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    obstacle_upper_thresholds = [1, 0.0001, 1, 1, 1, 1, 1, 1, 0.0002, 1, 1, 1]
    object_perminances = [100, 200, 100, 30, 30, 100, 30, 100, 100, 100, 100, 100]
    start_delays = [0, 0, 0, 1000, 500, 0, 0, 0, 1000, 1000, 5000, 1000]
    diff_thresholds = [60, 60, 60, 60, 60, 40, 40, 60, 50, 60, 60, 60]
    sudden_global_deltas = [4, 4, 4, 4, 4, 1, 2, 4, 4, 2, 2, 2, 2]
    human_occupancy = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    human_occupancy_thresholds = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 , 100]
elif DATA_SET == "Safety":
    # Safety's Parameters
    #videos = ["20230203_093758-2.mp4", "20230203_190436_trim.mp4", "20230203_182727-2.mp4","20230203_172727-2.mp4"]
    videos = ["safety_1/try" + str(i) + ".mp4" for i in range(1, 6)]
    start_frames = [18000, 6000, 30000, 0]
    buffer_scales = [1, 1, 1, 2]
    short_buffer_sizes = [100, 100, 100, 150]
    long_buffer_sizes= [200, 200, 200, 250]
    sample_rates = [25, 25, 25, 50]
    obstacle_thresholds = [0.001, 0.001, 0.01, 0.01]
    obstacle_upper_thresholds = [1, 1, 1, 1, 1]
    object_perminances = [500, 500, 500, 1500]
    start_delays = [0, 0, 0 ,0]
    diff_thresholds = [60, 60, 60, 60]
    sudden_global_deltas = [100, 100, 100, 100, 6, 4, 4, 4, 4]
    human_occupancy = [False, False, False, False, False, False, False, False, False]
    human_occupancy_thresholds = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
elif DATA_SET == "PETS":
    # Abandoned object detection paramete     
    videos = ["PETS/PET_S1_VIDEO/project3.avi", "PETS/PET_S1_VIDEO/project4.avi", "PETS/PET_S1_VIDEO/project1.avi","PETS/PET_S1_VIDEO/project2.avi",
              "PETS/PET_S2_VIDEO/project1.avi", "PETS/PET_S2_VIDEO/project2.avi", "PETS/PET_S2_VIDEO/project3.avi", "PETS/PET_S2_VIDEO/project4.avi",
              "PETS/PET_S3_VIDEO/project1.avi","PETS/PET_S3_VIDEO/project2.avi", "PETS/PET_S3_VIDEO/project3.avi", "PETS/PET_S3_VIDEO/project4.avi",
              "PETS/PET_S4_VIDEO/project1.avi","PETS/PET_S4_VIDEO/project2.avi", "PETS/PET_S4_VIDEO/project3.avi", "PETS/PET_S4_VIDEO/project4.avi",
              "PETS/PET_S5_VIDEO/project1.avi","PETS/PET_S5_VIDEO/project2.avi", "PETS/PET_S5_VIDEO/project3.avi", "PETS/PET_S5_VIDEO/project4.avi",
              "PETS/PET_S6_VIDEO/project1.avi","PETS/PET_S6_VIDEO/project2.avi", "PETS/PET_S6_VIDEO/project3.avi", "PETS/PET_S6_VIDEO/project4.avi",
              "PETS/PET_S7_VIDEO/project1.avi","PETS/PET_S7_VIDEO/project2.avi", "PETS/PET_S7_VIDEO/project3.avi", "PETS/PET_S7_VIDEO/project4.avi"]
    start_frames = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    buffer_scales = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    short_buffer_sizes = [250, 150, 250, 250, 300, 250, 300, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 375, 250, 250, 250, 250, 250]
    long_buffer_sizes= [450, 350, 350, 350, 400, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 525, 350, 350, 350, 350, 350]
    sample_rates = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  
    obstacle_thresholds = [0.0005, 0.0004, 0.00015, 0.00006, 0.00015, 0.00006, 0.0001, 0.00001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0002, 0.0002]
    obstacle_upper_thresholds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    object_perminances = [100, 400, 100, 10000, 100, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000]
    start_delays = [0, 0, 0, 0, 300, 1200, 300, 600, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 1000, 0, 1000, 500]
    diff_thresholds = [50, 40, 50, 50, 50, 40, 50, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 50, 50, 50, 30, 50, 50, 50, 40, 45]
    sudden_global_deltas = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    human_occupancy = [False, False, True, True, False, True, False, False, False, False, False, True , False, False, True, True, True, True, False, True, True, True, False, True, True, True, True, True]
    human_occupancy_thresholds = [100, 100, 100, 80, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    exclusion_mask_thresh = {'16': 0.2}

# Initialization
video_path = videos[video_index]
start_frame = start_frames[video_index]
short_buffer_size = int(short_buffer_sizes[video_index] * buffer_scales[video_index] )
long_buffer_size = int(long_buffer_sizes[video_index] * buffer_scales[video_index] )
sample_rate = sample_rates[video_index]
obstacle_thresh = obstacle_thresholds[video_index]
obstacle_upper_thresh = obstacle_upper_thresholds[video_index]
object_perminance = object_perminances[video_index]
start_delay = start_delays[video_index]
diff_thresh = diff_thresholds[video_index]
sudden_global_delta = sudden_global_deltas[video_index]
use_human_occupancy = human_occupancy[video_index]
human_occupancy_threshold = human_occupancy_thresholds[video_index]

# Push a frame unto the buffer
def push_frame(buffer, frame, pos, max_size):
    buffer[pos] = frame.copy()
    if pos + 1 >= max_size: pos = 0
    return pos + 1


if __name__ == '__main__':

    # Open video file
    vid = cv2.VideoCapture(video_path)
    _, frame_start = vid.read()

    # Initialize
    long_buffer = np.zeros((long_buffer_size,frame_start.shape[0], frame_start.shape[1]), float)  
    short_buffer = np.zeros((short_buffer_size, frame_start.shape[0], frame_start.shape[1]), float)
    freeze_long_avg = np.zeros((frame_start.shape[0], frame_start.shape[1]), np.uint8)
    persons_mask = np.zeros((frame_start.shape[0], frame_start.shape[1]), np.int16)
    exclusion_mask = np.zeros((frame_start.shape[0], frame_start.shape[1]), np.uint8)
    long_buffer_pos = 0
    short_buffer_pos = 0
    long_buffer_full = False
    short_buffer_full = False
    obstacle_detected = False
    obstacles = []
    total_objects = 0  
    model = YOLO('yolov8n.pt')

    # Start processing frames
    frame_number = 1
    print("Filling buffers...")
    while True:
        if frame_number > start_delay: 
            success, frame = vid.read()
            if not success: 
                print("Total abandoned objects detected: " + str(total_objects))
                cv2.waitKey()
                break
        else:
            frame = frame_start.copy()
        frame_number += 1

        if frame_number >= start_frame:
            if frame_number % sample_rate == 0:

                # Convert frame to gray, calculate number of total pixels and apply smoothing for noise reduction
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                gray = hsv[:, :, 2]
                num_pixels = gray.shape[0] * gray.shape[1]
                gray = cv2.blur(gray, (3, 3))

                # Object detection
                if use_human_occupancy:
                    results = model.predict(source=frame, save=False, save_txt=False, classes=[0])
                    if frame_number % (sample_rate * 2) == 0: 
                        # Decrease the confidence of all person occupied regions (decay the signal)
                        persons_mask -= 1
                        persons_mask[persons_mask < 0] = 0
                        
                    # Create a mask of persons detected
                    # Each person mask has time decay that is delayed by new detections 
                    for result in results:
                        boxes = result.boxes  # Boxes object for bbox outputs
                        masks = result.masks  # Masks object for segmentation masks outputs
                        keypoints = result.keypoints  # Keypoints object for pose outputs
                        probs = result.probs  # Class probabilities for classification outputs
                        for box in boxes:
                            b = np.int16(np.array(box.xyxy[0]))
                            persons_mask[b[1]:b[3], b[0]:b[2]] += 2 # Increase the confidence of person occupied regions of mask
                    persons_mask[persons_mask > 255] = 255
                    exclusion_mask = np.uint8(persons_mask.copy())
                    exclusion_mask[exclusion_mask > human_occupancy_threshold] = 255
                    exclusion_mask[exclusion_mask <= human_occupancy_threshold] = 0


                # Add to the buffers
                long_buffer_pos = push_frame(long_buffer, gray, long_buffer_pos, long_buffer_size)
                short_buffer_pos = push_frame(short_buffer, gray, short_buffer_pos, short_buffer_size)
                if long_buffer_pos >= long_buffer_size-1: long_buffer_full = True
                if short_buffer_pos >= short_buffer_size-1: short_buffer_full = True

                if long_buffer_full:
                    # Create reference average frames
                    long_avg_frame = np.uint8(long_buffer.sum(axis=0)/long_buffer_size)
                    short_avg_frame = np.uint8(short_buffer.sum(axis=0)/short_buffer_size)                   
                    if not obstacle_detected: freeze_long_avg = long_avg_frame.copy()

                    # Take difference of buffers and threshold
                    long_diff = cv2.absdiff(long_avg_frame, gray)
                    short_diff = cv2.absdiff(short_avg_frame, gray)
                    ret, long_diff_thresh =  cv2.threshold(long_diff, diff_thresh, 255, cv2.THRESH_BINARY)
                    ret, short_diff_thresh = cv2.threshold(short_diff, diff_thresh, 255, cv2.THRESH_BINARY)

                    # Detect obstacle change
                    kernel = np.ones((3,3),np.uint8)
                    obstacle_frame = cv2.morphologyEx(cv2.absdiff(long_diff_thresh, short_diff_thresh), cv2.MORPH_CLOSE, kernel)
                    if obstacle_frame.sum()/num_pixels > sudden_global_delta: # If significant changes in ambient light, reset buffers
                        print("BUFFER RESET -", obstacle_frame.sum()/num_pixels)
                        for i in range(long_buffer.shape[0]): long_buffer[i] = gray
                        for i in range(short_buffer.shape[0]): short_buffer[i] = gray
                        obstacle_frame = np.zeros(gray.shape, np.uint8)                        
                    contours, _ = cv2.findContours(obstacle_frame,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    obstacle_frame = cv2.cvtColor(obstacle_frame, cv2.COLOR_GRAY2BGR)
                    if len(contours) > 0:
                        contour_sizes = [cv2.contourArea(cnt) for cnt in contours]

                        # Store all obstacles that are large enough
                        oBuff = 50  # Overlap buffer to consider two obstacles as one
                        for contour in contours:
                            area = cv2.contourArea(contour) # Get the area of the blob
                            x,y,w,h = cv2.boundingRect(contour) # Get the positin of the blob
                            rel_area = area/num_pixels
                            if rel_area > obstacle_thresh * 0.25: # object diff has to be a quarter of the acceptable minimum size (used to detect when object is removed with more sensitivity)

                                # Look through past obstacles to find a match
                                obstacle_found = False
                                for i, obstacle in enumerate(obstacles):
                                    ob_x,ob_y,ob_w,ob_h = cv2.boundingRect(obstacle["contour"])
                                    ob_area = cv2.contourArea(obstacle["contour"])
                                    buff = np.zeros(gray.shape, np.uint8)   # This is used to detect if new object overlaps existing obstacle
                                    buff[ob_y-oBuff:ob_y+ob_h+oBuff, ob_x-oBuff:ob_x+ob_w+oBuff] = 1
                                    if np.sum(buff[y:y+h, x:x+w]) > 0:  # Overlap was found
                                        obstacle_found = True
                                        break

                                # Does obstacle overlap with human occupied region
                                overlap_with_human = np.sum(exclusion_mask[y:y+h, x:x+w] > 0)/(w*h) > .5 if not str(video_index) in exclusion_mask_thresh else exclusion_mask_thresh[str(video_index)]
                                if np.sum(exclusion_mask[y:y+h, x:x+w] > 0)/(w*h)  > 0: print(np.sum(exclusion_mask[y:y+h, x:x+w] > 0)/(w*h))

                                # Obstacle is found if there is no overlap with existing obstacle
                                if not obstacle_found and  obstacle_upper_thresh > rel_area > obstacle_thresh: 
                                    if not overlap_with_human:
                                        obstacle = {"contour": contour, 
                                                    "last_frame": frame_number, 
                                                    "update_count": 0,
                                                    "rect": np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape((-1, 1, 2)),
                                                    "img_region": (y,y+h, x, x+w) }
                                        obstacles.append(obstacle)
                                elif  obstacle_found and (frame_number - obstacles[i]["last_frame"] > object_perminance or overlap_with_human):
                                    #print(frame_number - obstacles[i]["last_frame"] )
                                    for j in range(len(short_buffer)): short_buffer[j,ob_y:ob_y+ob_h,ob_x:ob_x+ob_w] = gray[ob_y:ob_y+ob_h,ob_x:ob_x+ob_w]
                                    for j in range(len(long_buffer)): long_buffer[j,ob_y:ob_y+ob_h,ob_x:ob_x+ob_w] = gray[ob_y:ob_y+ob_h,ob_x:ob_x+ob_w]
                                    obstacles.pop(i)                          
                                elif  obstacle_upper_thresh > rel_area > obstacle_thresh:
                                    #obstacle_frames[i] = frame_number
                                    obstacles[i]["last_frame"] = frame_number
                                    if area > ob_area: 
                                        #obstacles[i] = contour
                                        obstacles[i]["contour"] = contour
                                        #obstacle_rects[i] = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape((-1, 1, 2))       
                                        obstacles[i]["rect"] = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape((-1, 1, 2)) 
                                        obstacles[i]["update_count"] += 1                            
                            
                        cv2.drawContours(image=obstacle_frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                        #cv2.putText(obstacle_frame, 'Largest Contour:  ' + str(contour_sizes[largest_contour]/(gray.shape[0] * gray.shape[1])), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)
                    elif not obstacle_detected:
                        T = 0

                    # Visualization
                    cv2.imshow('Long Buffer', long_avg_frame)
                    cv2.moveWindow('Long Buffer', 400, 0)
                    cv2.imshow('Short Buffer', short_avg_frame)
                    cv2.moveWindow('Short Buffer', 800, 0)
                    cv2.imshow('Long Difference', long_diff_thresh)
                    cv2.moveWindow('Long Difference', 400, 600)
                    cv2.imshow('Short Difference', short_diff_thresh)
                    cv2.moveWindow('Short Difference', 800, 600)
                    cv2.imshow('Obstacle Frame', obstacle_frame)
                    cv2.moveWindow('Obstacle Frame', 0, 600)
                    

                # Visualize output
                object_found = False          
                numObjects = 0   
                if len(obstacles) > 0: 
                    for i, obstacle in enumerate(obstacles):                        
                        if obstacle["update_count"] > 3:
                            object_found = True
                            numObjects += 1
                            cv2.drawContours(image=frame, contours=[obstacle["contour"]], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                            cv2.polylines(frame, [obstacle["rect"]], True, (0, 0, 255), 2)
                            #cv2.imshow("Object " + str(i), frame[obstacle["img_region"][0]:obstacle["img_region"][1], obstacle["img_region"][2]:obstacle["img_region"][3]])
                if numObjects > total_objects: total_objects = numObjects

                # Draw outline of person mask
                ex_contours, _ = cv2.findContours(exclusion_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(ex_contours) > 0: cv2.drawContours(image=frame, contours=ex_contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                
                cv2.putText(frame, 'Frame ' + str(frame_number), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if object_found: cv2.putText(frame, 'Abandoned Object(s): ' + str(total_objects), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Original', frame)
                cv2.moveWindow('Original',0, 0)
                if cv2.waitKey(1) & 0xff == ord('q'): break



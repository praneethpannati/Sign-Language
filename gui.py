import streamlit as st
from gtts import gTTS
import numpy as np
from keras.models import load_model
from mediapipe import solutions
import os
import cv2
import time

curr_location=os.path.dirname(os.path.abspath(__file__))
model_location="Conv_Lstm.h5"

def videoLabels():
    return {0: 'loud',1: 'quiet',2: 'happy',3: 'long',4: 'short',5: 'large',6: 'little',7: 'hot',8: 'new',9: 'good',10: 'dry',
 11: 'Red',12: 'Black',13: 'White',14: 'Monday',15: 'Year',16: 'Time',17: 'Window',18: 'Pen',19: 'Paint',20: 'Teacher',
 21: 'Priest',22: 'Car',23: 'ticket',24: 'Father',25: 'Brother',26: 'Boy',27: 'Girl',28: 'House',29: 'Court',30: 'Shop',
 31: 'Bank',32: 'Election',33: 'Death'}

def is_valid_path(path):
    """Check if the given path location is a valid directory or file."""
    return os.path.exists(path) and (os.path.isdir(path) or os.path.isfile(path))

def play_video(video_path):
    """Play a video file."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return
    # Loop through the frames and display them
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of the video.")
            break
        cv2.imshow("Video", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def preprocessVideo(path,nfsize=25,sz=150):
    mp_drawing=solutions.drawing_utils
    mpDraw=solutions.drawing_utils
    mp_pose=solutions.pose
    mpPose=solutions.pose
    pose = mpPose.Pose()
    mpHands = solutions.hands
    hands = mpHands.Hands()
    x=[]
    size=nfsize
    count=-1
    # iterating through the video
    cap = cv2.VideoCapture(path) # opening the video
    if cap.isOpened():  # checking if the video is opened
        nfs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//size # calculating the number of frames to be skipped
        if nfs==0: # if the number of frames to be skipped is 0
            nfs=1 # setting the number of frames to be skipped to 1
        x1,x2,x3=[],[],[] # creating the empty lists
        fps = (cap.get(cv2.CAP_PROP_FPS)) # getting the fps of the video
        while True: # iterating through the frames
            ret,frame = cap.read()
            if not ret: # if the frame is not read
                break # break the loop
            # resizing the frame
            frame=cv2.resize(frame,(600,600))
            img1=np.zeros(frame.shape)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for handlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img1, handlms, mpHands.HAND_CONNECTIONS)
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Render detections
            mp_drawing.draw_landmarks(img1, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            # storing img1 in frame variable
            frame=img1
            # getting the frame count
            frmid=cap.get(1)
            # resizing the frame
            frame=cv2.resize(frame,(sz,sz))
            # appending the frame to the list
            if frmid%nfs==0 and ret:
                x1.append(frame/255)
                x2.append(np.array(cv2.flip(frame,1))/255) # flipping the frame

    # appending the frames to the list
    if len(x1)!=size:
        if len(x1)<size:
            x1.append(x1[-(size-len(x1))])
            x2.append(x2[-(size-len(x1))])
        else:
            x1=x1[0:size]
            x2=x2[0:size]
    x.append(x1)
    x.append(x2)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return np.array(x)

def loadOurModel(selected_model_path):
    """Loads model"""
    return load_model(selected_model_path)

def predictSign(myModel,preprocessed_video):
    pred=np.argmax(myModel.predict(preprocessed_video),axis=1)
    OurSignLabels=videoLabels()
    return [OurSignLabels[x] for x in pred]

def our_prediction(selected_video_path,selected_model_path):
    if selected_video_path[0]=="\"" and selected_video_path[-1]=="\"":
        selected_video_path=selected_video_path[1:-1]
    
    if is_valid_path(selected_video_path):
    #     if os.path.isdir(selected_video_path):
    #         print(f"The path '{selected_video_path}' is a valid directory.")
    #     elif os.path.isfile(selected_video_path):
    #         print(f"The path '{selected_video_path}' is a valid file.")
    
        # play_video(selected_video_path)
        start_time = time.time()
        pred=predictSign(loadOurModel(selected_model_path),preprocessVideo(selected_video_path))
        end_time = time.time()
        print(pred)
        print(f"OUR PREDICTION IS:{pred[0]}")
    else:
        print(f"The path '{selected_video_path}' is not valid.")
        return ["Invalid",0]
    return [pred[0],end_time-start_time]




# Load models
model_path_dict = {'Conv_Lstm': model_location,}
model_names_list=['Conv_Lstm',]



## Streamlit App
st.markdown("<h1 style='text-align: center; color: black;'>Sign Language Translator</h1>", unsafe_allow_html=True)


# Create two columns
col1, col2 ,col3 = st.columns(3)
with col1:
    # Dropdown menu for model selection
    selected_model = st.selectbox("Our Model:", list(model_path_dict.keys()))
    # selected_model = st.selectbox("Our Model:", model_names_list)
    # selected_model_path = os.path.join(curr_location, selected_model+".keras")
    # Image upload function
    uploaded_file = st.file_uploader("Input Video", type=["MOV", "MP4","mp4", "avi"])


if uploaded_file is not None:
    # Perform segmentation using the selected model
    # st.write("Selected Video:", uploaded_file.name)
    # os.makedirs(curr_location, exist_ok=True)
    video_path = os.path.join(curr_location, uploaded_file.name)
    # Save the uploaded file to the specified location
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        [prediction,total_time]=our_prediction(video_path,model_path_dict[selected_model])
        # ans=our_prediction(video_path,selected_model_path)
        os.remove(video_path)
    except:
        os.remove(video_path)
    # prediction=our_prediction(video_path,model_path_dict[selected_model])
    # os.remove(video_path)
    # st.header(f":black[Predicted Sign : {prediction}]")

    with col2:
        with st.container(height=450,border=False):
            st.write("<font color='black'><h3>Video Player</h3></font>",unsafe_allow_html=True)
            # Display the video player
            st.video(uploaded_file)
            st.write(f"<font color='black'>Overall Time taken to predict : {total_time-3:.4f}</font>",unsafe_allow_html=True)
    
    
    with col3:
        # st.write(f"<font color='black'><h3>Predicted Sign : {ans}</h3></font>",unsafe_allow_html=True)
        st.header(f":black[Predicted Sign : {prediction}]")
        st.subheader(":black[Audio]")
        audio = gTTS(text=prediction, lang='en', slow=False)
        # Save audio to a temporary file
        # os.makedirs(curr_location, exist_ok=True)
        audio_file_path = os.path.join(curr_location, "test_temp_audio.mp3")
        audio.save(audio_file_path)
        
        # Display audio player
        try:
            st.audio(audio_file_path)
            # st.write(f'<audio src="{audio_file_path}" id="audio" autoplay="autoplay" controls="controls">', unsafe_allow_html=True)
            # st.script("""var audio = document.getElementById("audio");audio.play();""")
            os.remove(audio_file_path)
        except:
            os.remove(audio_file_path)
        # space between colums
        # st.write('<style>div.row-widget.stHorizontal { flex-wrap: nowrap; }</style>', unsafe_allow_html=True)
        

# Run the app


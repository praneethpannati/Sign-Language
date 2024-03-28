import streamlit as st
from gtts import gTTS
import numpy as np
from keras.models import load_model
from mediapipe import solutions
import os
import cv2
# import base64

curr_location=os.path.dirname(os.path.abspath(__file__))
model_location=r"C:\Users\palna\FYP\conv_lstm2 (1).keras"

def videoLabels():
    return {0: 'loud',1: 'quiet',2: 'happy',3: 'long',4: 'short',5: 'large',6: 'little',7: 'hot',8: 'new',9: 'good',10: 'dry',
 11: 'Red',12: 'Black',13: 'White',14: 'Monday',15: 'Year',16: 'Time',17: 'Window',18: 'Pen',19: 'Paint',20: 'Teacher',
 21: 'Priest',22: 'Car',23: 'ticket',24: 'Father',25: 'Brother',26: 'Boy',27: 'Girl',28: 'House',29: 'Court',30: 'Shop',
 31: 'Bank',32: 'Election',33: 'Death'}

def is_valid_path(path):
    """Check if the given path location is a valid directory or file."""
    return os.path.exists(path) and (os.path.isdir(path) or os.path.isfile(path))

# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_background(png_file):
#     bin_str = get_base64(png_file)
#     st.markdown('''<style>.stApp {background-image: url("data:image/png;base64,%s");background-size: cover;}</style>''' % bin_str, unsafe_allow_html=True)

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

def loadOurModel(selected_model_loc):
    """Loads model"""
    return load_model(selected_model_loc)

def predictSign(myModel,preprocessed_video):
    pred=np.argmax(myModel.predict(preprocessed_video),axis=1)
    OurSignLabels=videoLabels()
    return [OurSignLabels[x] for x in pred]

def our_prediction(input_path,selected_model_loc):
    print(" IM INSIDE ")
    if input_path[0]=="\"" and input_path[-1]=="\"":
        input_path=input_path[1:-1]
    
    if is_valid_path(input_path):
        if os.path.isdir(input_path):
            print(f"The path '{input_path}' is a valid directory.")
        elif os.path.isfile(input_path):
            print(f"The path '{input_path}' is a valid file.")
        
        x_val = preprocessVideo(input_path)
        # play_video(input_path)
        myModel=loadOurModel(selected_model_loc)
        pred = predictSign(myModel,x_val)
        print(pred)
    else:
        print(f"The path '{input_path}' is not valid.")
        return "Invalid"
    print(pred[0])
    return pred[0]




# Load models
model_path_dict = {'Conv_Lstm': model_location,}
model_names_list=['Conv_Lstm',]



## Streamlit App
st.markdown("<h1 style='text-align: center; color: black;'>Sign Language Translator</h1>", unsafe_allow_html=True)
# set_background(r"C:\Users\rmsre\Documents\Python Scripts\SLR\ISL.png")
ans=""

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
        
        ans = our_prediction(video_path,model_path_dict[selected_model])
        # ans=our_prediction(video_path,selected_model_path)
        os.remove(video_path)
    except:
        os.remove(video_path)
    

    with col2:
        with st.container(height=450,border=False):
            st.write("<font color='black'><h3>Video Player</h3></font>",unsafe_allow_html=True)
            # Display the video player
            st.video(uploaded_file)
    
    
    with col3:
        # st.write(f"<font color='black'><h3>Predicted Sign : {ans}</h3></font>",unsafe_allow_html=True)
        
        st.header(f":black[Predicted Sign : {ans}]")

        st.subheader(":black[Audio]")
        audio = gTTS(text = ans, lang='en', slow=False)
        # Save audio to a temporary file
        # os.makedirs(curr_location, exist_ok=True)
        audio_path = os.path.join(curr_location, "test_temp_audio.mp3")
        audio.save(audio_path)
        
        # Display audio player
        try:
            st.audio(audio_path)
            os.remove(audio_path)
        except:
            os.remove(audio_path)
    
        # space between colums
        # st.write('<style>div.row-widget.stHorizontal { flex-wrap: nowrap; }</style>', unsafe_allow_html=True)
        

# Run the app

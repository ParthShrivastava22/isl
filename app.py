import streamlit as st
import cv2
import joblib
import numpy as np
import tempfile

# --- 1. Hardcoded Class Names (Extracted directly from Notebook output) ---
RAW_CLASSES = [
    'O.mp4', 'M.mp4', 'Name.mp4', 'Before.mp4', 'Keep.mp4', '5.mp4', 'Thank.mp4', 'Happy.mp4', 
    'Do.mp4', 'From.mp4', 'Sound.mp4', 'Our.mp4', 'Pretty.mp4', 'Distance.mp4', 'U.mp4', 'S.mp4', 
    'On.mp4', 'This.mp4', 'H.mp4', 'Not.mp4', 'Glitter.mp4', 'Home.mp4', 'Do', 'At.mp4', 
    'Yourself.mp4', 'C.mp4', 'When.mp4', '0.mp4', 'Hand.mp4', 'Age.mp4', '1.mp4', 'Y.mp4', 
    'Cannot.mp4', 'Does', 'Alone.mp4', 'Walk.mp4', '2.mp4', 'Wash.mp4', 'Out.mp4', 'Thank', 
    'ME.mp4', 'Computer.mp4', 'Fight.mp4', 'Which.mp4', 'Why.mp4', 'Of.mp4', 'Language.mp4', 
    'Laugh.mp4', '6.mp4', 'Next.mp4', 'But.mp4', 'Help.mp4', 'Can.mp4', 'Sad.mp4', 'R.mp4', 
    'God.mp4', 'World.mp4', 'So.mp4', 'I.mp4', 'Your.mp4', 'Study.mp4', 'G.mp4', 'F.mp4', 
    'Her.mp4', 'To.mp4', 'Sign.mp4', 'L.mp4', 'Will.mp4', 'Ask.mp4', 'Bye.mp4', 'Self.mp4', 
    'Go.mp4', 'D.mp4', 'That.mp4', 'See.mp4', '8.mp4', 'Without.mp4', 'Better.mp4', 'Work.mp4', 
    '4.mp4', 'Whole.mp4', 'Stay.mp4', 'Television.mp4', 'His.mp4', 'Invent.mp4', 'What.mp4', 
    'Eat.mp4', 'B.mp4', 'My.mp4', 'And.mp4', 'Time.mp4', 'Where.mp4', 'They.mp4', 'Now.mp4', 
    'Q.mp4', 'W.mp4', 'It.mp4', 'V.mp4', '3.mp4', 'Safe.mp4', 'Best.mp4', 'How.mp4', 'Change.mp4', 
    'Type.mp4', 'J.mp4', 'Gold.mp4', 'Welcome.mp4', 'Sing.mp4', '7.mp4', 'Hello.mp4', 'Talk.mp4', 
    'All.mp4', 'Great.mp4', 'Wrong.mp4', 'Busy.mp4', 'Learn.mp4', 'Z.mp4', 'Again.mp4', 'Whose.mp4', 
    'Come.mp4', 'Way.mp4', 'K.mp4', 'Right.mp4', 'Who.mp4', 'E.mp4', 'Homepage.mp4', 'Here.mp4', 
    'Beautiful.mp4', '9.mp4', 'Us.mp4', 'Engineer.mp4', 'Those.mp4', 'P.mp4', 'Finish.mp4', 'You.mp4', 
    'Day.mp4', 'With.mp4', 'Also.mp4', 'After.mp4', 'Be.mp4', 'Words.mp4', 'X.mp4', 'More.mp4', 
    'Hands.mp4', 'Against.mp4', 'T.mp4', 'We.mp4', 'Good.mp4', 'N.mp4', 'A.mp4', 'College.mp4'
]

# Clean up the .mp4 extensions and create a dictionary mapping index to word
CLASS_NAMES = {i: name.replace('.mp4', '') for i, name in enumerate(RAW_CLASSES)}

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('isl_video_model.pkl')

model = load_model()

# --- 3. Build the UI ---
st.title("Indian Sign Language Classifier 🤟")
st.write("Upload an ISL video file, and the model will predict the sign based on the first frame.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    ret, frame = cap.read()
    cap.release()

    if ret:
        st.write("Extracting the first frame for prediction...")
        
        # Display the frame to the user (OpenCV uses BGR, Streamlit needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Analyzed Frame", width=350)

        # --- 4. Preprocess and Predict ---
        # Match the exact preprocessing from the Jupyter notebook (Resize to 32x32 and flatten)
        feat = cv2.resize(frame, (32, 32)).flatten()
        feat = feat.reshape(1, -1) # Reshape for a single prediction
        
        prediction_index = model.predict(feat)[0]
        predicted_sign = CLASS_NAMES.get(prediction_index, "Unknown Sign")

        st.success(f"### Predicted Sign: **{predicted_sign}**")
    else:
        st.error("Could not read the video file. Please try another one.")
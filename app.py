import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# ฟังก์ชันโหลดไฟล์ .pkl
def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

# โหลดโมเดล
rf_class_model = load_pickle("model_rf_class.pkl")
rf_branch_model = load_pickle("model_rf_branch.pkl")

# โหลด Encoders
class_encoder = load_pickle("class_encoder.pkl")
branch_encoder = load_pickle("branch_encoder.pkl")

# โหลด Scaler
scaler = load_pickle("scaler.pkl")

# โหลด Dataset Arknights
data = pd.read_csv("Arknights_Dataset.csv", encoding="latin-1")

# คำอธิบายฟีเจอร์
column_descriptions = {
    "Name": "ชื่อตัวละคร",
    "ATK": "ค่าพลังโจมตี",
    "HP": "ค่าพลังชีวิต",
    "DEF": "ค่าพลังป้องกัน",
    "RES": "ค่าต้านทานอาร์ต(ดาเมจเวท)",
    "Redeployment time": "ระยะเวลาในการลงสนามใหม่อีกครั้ง",
    "DP cost": "ค่า DP ที่ต้องใช้เพื่อลงสนาม",
    "Block count": "จำนวนบล็อกสูงสุดของตัวละคร",
    "Attack interval": "ความเร็วโจมตีของตัวละคร",
    "Position": "ตำแหน่งการยืนของตัวละคร",
    "Rarity": "ระดับดาวของตัวละคร",
    "Class": "คลาสของตัวละคร",
    "Branch": "สายของตัวละคร"
}

# แปลงเป็น DataFrame
feature_df = pd.DataFrame(list(column_descriptions.items()), columns=["Feature", "Description"])

# ฟังก์ชันอัปเดต LabelEncoder
def update_label_encoder(encoder, column):
    unique_values = data[column].dropna().astype(str).unique() 
    encoder_classes = encoder.classes_.astype(str)
    
    encoder.classes_ = np.unique(np.concatenate((encoder_classes, unique_values)))
    return encoder

# โหลดและอัปเดต LabelEncoders
class_encoder = pickle.load(open("class_encoder.pkl", "rb"))
branch_encoder = pickle.load(open("branch_encoder.pkl", "rb"))

class_encoder = update_label_encoder(class_encoder, "Class")
branch_encoder = update_label_encoder(branch_encoder, "Branch")

#**************************************************************************************************************************************************#

# กำหนด path ของโฟลเดอร์ dataset
dataset_folder = "Hairstyles"

# โหลดชื่อทรงผมจากโฟลเดอร์หลัก
if os.path.exists(dataset_folder):
    hairstyle_classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
else:
    hairstyle_classes = []

# โหลดภาพทั้งหมดใน dataset พร้อมกับ label
image_paths = []
for hairstyle in hairstyle_classes:
    folder_path = os.path.join(dataset_folder, hairstyle)
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths.extend(image_files)

# สร้าง Encoder และบันทึก
hairstyle_encoder = LabelEncoder()
if len(hairstyle_classes) > 0:  # ป้องกัน error
    hairstyle_encoder.fit(hairstyle_classes)

encoder_path = "hairstyle_encoder.pkl"
with open(encoder_path, "wb") as f:
    pickle.dump(hairstyle_encoder, f)

# โหลดโมเดล Neural Network
nn_model = load_model("neural_model.h5", compile=False)

# ฟังก์ชันประมวลผลรูปภาพ
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # แปลงเป็นภาพขาวดำ
    image = cv2.resize(image, (128, 128))  # ปรับขนาด
    image = np.expand_dims(image, axis=-1)  # เพิ่มมิติให้เป็น (128, 128, 1)
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้โมเดลอ่านได้ (1, 128, 128, 1)
    return image

#**************************************************************************************************************************************************#

# Sidebar Navigation
st.sidebar.title("Navigation Bar")
page = st.sidebar.radio("", [
    "Arknights Class and Branch Prediction (ML Model)",
    "Arknights Class and Branch Prediction (Demo)",
    "Anime Girl Hairstyle Prediction (Neural Network Model)",
    "Anime Girl Hairstyle Prediction (Demo)"
])

# หน้าแรก - อธิบายโมเดล
if page == "Arknights Class and Branch Prediction (ML Model)":
    
    col1, col2 = st.columns([3, 3]) 

    with col1:
        st.title("Machine Learning Model Detail")
    with col2:
        st.image("ML/arknights_logo.png", width=400)  
    
    st.markdown("""

---

### Introduction to Arknights   
Arknights เป็นเกมแนว Tower Defense + Strategy RPG ที่พัฒนาโดย Hypergryph และให้บริการโดย Yostar เปิดตัวครั้งแรกในประเทศจีนเมื่อปี 2019 ก่อนจะเปิดให้เล่นทั่วโลกในปี 2020
\nในเกมนี้ ผู้เล่นจะรับบทเป็น Doctor ผู้นำขององค์กร Rhodes Island, องค์กรทางการแพทย์และทหารรับจ้างที่ต่อสู้กับ Originium Infection, โรคปริศนาที่แพร่กระจายไปทั่วโลก Terra
                
---

### Model ทำนาย Class และ Branch ของ Operators ในเกม Arknights โดยใช้ Stats ของตัวละคร

### แหล่งข้อมูลของ Dataset: Arknights Terra Wiki

**Dataset Details:**
- ประกอบด้วยข้อมูลสเตตัสของ Operator ในเกม Arknights ทั้งหมด 336 records
- ข้อมูลเหล่านี้ถูกนำมาใช้ Train โมเดล เพื่อทำนาย Class และ Branch ของ Operators""")

    st.table(feature_df)

    st.markdown("""

---
                
### โมเดลที่ใช้ Train

### Random Forest Classifier (โมเดลที่ใช้ใน Demo)
\nRandom Forest เป็น ensemble learning method ที่ใช้ หลายๆ ต้นไม้การตัดสินใจ (Decision Trees) มาช่วยกันพยากรณ์ผลลัพธ์ ซึ่งช่วยลดปัญหา overfitting และเพิ่มความแม่นยำ

หลักการทำงานของ Random Forest
- สร้างต้นไม้ (Decision Trees) จากตัวอย่างข้อมูลที่สุ่มขึ้นมา (Bootstrap Sampling)
- ต้นไม้แต่ละต้น จะพยากรณ์ผลลัพธ์ของตัวเอง
- ใช้ การโหวตเสียงข้างมาก (Majority Voting) ในกรณีของ Classification
- หรือใช้ ค่าเฉลี่ย (Averaging) ในกรณีของ Regression
                
ข้อดีของ Random Forest
- ลด overfitting ที่อาจเกิดขึ้นใน Decision Tree
- มีความแม่นยำสูง โดยเฉพาะกับข้อมูลที่มี noise
- สามารถจัดการกับข้อมูลที่มี feature จำนวนมาก ได้ดี
- รองรับข้อมูลที่ ขาดหายไปบางส่วน (Missing Data)
                
ข้อเสียของ Random Forest
- ใช้ทรัพยากรสูง และช้า เมื่อมีจำนวนต้นไม้เยอะๆ
- ยากต่อการตีความ (Interpretability) เพราะใช้หลายต้นไม้

---

### Support Vector Machine (SVM)
\nSupport Vector Machine (SVM) เป็น โมเดลที่ใช้เส้นแบ่งข้อมูล (Hyperplane) เพื่อแยกคลาส โดยพยายามหาเส้นที่มี ระยะห่าง (Margin) มากที่สุด ระหว่างคลาส

หลักการทำงานของ SVM
- หาเส้นแบ่งข้อมูล (Hyperplane) ที่ดีที่สุด ซึ่งทำให้ margin กว้างที่สุด
- ใช้ Support Vectors (จุดข้อมูลที่อยู่ใกล้เส้นแบ่งมากที่สุด) เพื่อกำหนดตำแหน่งของ Hyperplane
- ถ้าข้อมูลไม่สามารถแยกกันได้แบบเส้นตรง (Linear Separable) จะใช้ Kernel Trick เพื่อแปลงข้อมูลไปอยู่ในมิติที่สูงขึ้น

ข้อดีของ SVM
- มีความแม่นยำสูง โดยเฉพาะกับข้อมูลที่มีขนาดเล็ก
- มีความสามารถในการแยกข้อมูลที่ซับซ้อนผ่าน Kernel Trick
- ทำงานได้ดีใน high-dimensional space
                
ข้อเสียของ SVM
- ใช้เวลาในการ train นาน เมื่อ dataset มีขนาดใหญ่
- การเลือก ค่า Hyperparameter เช่น C และ Gamma มีผลต่อประสิทธิภาพของโมเดล
- การตีความ (Interpretability) ค่อนข้างยาก
                
---

### อธิบายโค้ด      

""")
    
    st.write("Import Library ที่จำเป็น:")
    st.image("ML/import.png", width=700)

    st.write("อัปโหลดข้อมูล (Dataset) ขึ้น Colab:")
    st.image("ML/upload_data.png", width=700)

    st.write("โหลดข้อมูลจาก Dataset:")
    st.image("ML/load_data.png", width=700)

    st.write("กำหนดวิธีการจัดการกับ Missing Value แต่ละค่า:")
    st.image("ML/handle_missing.png", width=700)
    st.write("วิธีการจัดการ:")
    st.image("ML/use_mean.png", caption="ใช้ค่า Mean ในการจัดการ", width=700)
    st.image("ML/use_mode.png", caption="ใช้ค่า Mode ในการจัดการ", width=700)

    st.write("สร้าง Encoder สำหรับ Class และ Branch (เพื่อนำไปใช้ต่อ) และทำการ Encode column อื่นๆ:")
    st.image("ML/encoder.png", width=700)

    st.write("กำหนดฟีเจอร์และเป้าหมาย:")
    st.image("ML/define_feature.png", width=700)

    st.write("จัดการกับค่า NaN (ถ้ามี):")
    st.image("ML/handle_nan.png", width=700)

    st.write("Scaling (ใช้ Standard Scaler):")
    st.image("ML/scaling.png", width=700)

    st.write("Train Random Forest Model:")
    st.image("ML/rf_train.png", width=700)

    st.write("Train SVM Model:")
    st.image("ML/svm_train.png", width=700)

    st.write("แสดงผลลัพธ์ของการ Train:")
    st.image("ML/evaluate.png", width=700)
    st.image("ML/test_result.png", caption="ผลลัพธ์ของการ Train", width=700)

    st.write("บันทึกโมเดล และ Encoder:")
    st.image("ML/save_model.png", width=700)

    st.markdown("""---""")

    st.markdown("""### Demo: Arknights Class & Branch Prediction""")
    st.write("สำหรับ Demo ของ ML Model ผู้ใช้จะสามารถกดปุ่ม 'สุ่มตัวละคร' เพื่อสุ่มตัวละคร 1 ตัว จาก Dataset และโมเดลจะทำการทำนาย Class และ Branch ของตัวละครนั้น ดังตัวอย่าง: ")
    st.image("ML/arknights_demo.png", width=700)

    st.markdown("""
                
---
                
## แนวทางในการพัฒนาโมเดล
- Dataset ควรมีจำนวน records มากกว่านี้
- ลองใช้วิธีอื่นๆ ในการจัดการกับ Missing Value
- ลอง Train ด้วยโมเดลอื่นๆ
                
""")

#**************************************************************************************************************************************************#

# หน้าทดสอบพยากรณ์
elif page == "Arknights Class and Branch Prediction (Demo)":
    st.title("Arknights Class & Branch Prediction (Demo)")

    if st.button("🎲 สุ่มตัวละคร"):
        while True:
            random_idx = np.random.randint(0, len(data))
            sample = data.iloc[[random_idx]]

            # ตรวจสอบว่าข้อมูลไม่มีค่า Missing
            if sample.isnull().sum().sum() == 0:
                break

        # ดึงค่าที่แท้จริง (True Labels)
        y_true_class = sample["Class"].values[0]
        y_true_branch = sample["Branch"].values[0]

        # เตรียม Features โดยลบคอลัมน์ที่ไม่เกี่ยวข้อง
        X_sample = sample.drop(columns=["Name", "Class", "Branch"], errors="ignore")

        # แปลงข้อมูล Categorical -> ตัวเลข
        for col in X_sample.columns:
            if X_sample[col].dtype == "object":
                unique_values = list(class_encoder.classes_)  # ค่าเดิมที่ใช้ตอนเทรน
                new_values = X_sample[col].unique()  # ค่าที่สุ่มมาใหม่

                # ถ้ามีค่าที่ไม่รู้จัก ให้เพิ่มเข้าไปใน LabelEncoder และ fit ใหม่
                unseen_values = [val for val in new_values if val not in unique_values]
                if unseen_values:
                    class_encoder.classes_ = np.concatenate((class_encoder.classes_, unseen_values))

                # แปลงข้อมูลโดยไม่มี error
                X_sample[col] = class_encoder.transform(X_sample[col].astype(str))

        # ตรวจสอบให้แน่ใจว่ามีแต่ค่าตัวเลข
        X_sample = X_sample.select_dtypes(include=[np.number])

        # ทำ Scaling ข้อมูล
        X_sample = scaler.transform(X_sample)

        # ทำนายผลลัพธ์
        pred_class_rf = rf_class_model.predict(X_sample)[0]
        pred_branch_rf = rf_branch_model.predict(X_sample)[0]

        # แปลงค่าที่ทำนายกลับไปเป็น String เหมือนเดิม
        pred_class_rf = class_encoder.inverse_transform([pred_class_rf])[0]
        pred_branch_rf = branch_encoder.inverse_transform([pred_branch_rf])[0]

        # แสดงผล
        st.write("## 🎭 Sample Character Information")
        st.write(sample)

        st.write("### ✅ True Label")
        st.write(f"**Class:** {y_true_class}")
        st.write(f"**Branch:** {y_true_branch}")

        st.write("### 🤖 Model Prediction")
        st.write(f"**Class:** {pred_class_rf}")
        st.write(f"**Branch:** {pred_branch_rf}")

#**************************************************************************************************************************************************#

# หน้าอธิบายโมเดล Anime Hairstyle
elif page == "Anime Girl Hairstyle Prediction (Neural Network Model)":
    
    col1, col2 = st.columns([3, 3]) 

    with col1:
        st.title("Neural Network Model Detail")
    with col2:
        st.image("NN/Title.png", width=500)
    
    st.markdown("""
                
---
                
### Model จำแนกทรงผมตัวละครอนิเมะ(หญิง)จากรูปภาพ

### ที่มาของ Dataset: Deepai (Image Generator AI)

**Dataset Details:**
- ประกอบด้วยข้อมูลรูปภาพตัวละครอนิเมะ(หญิง) ในทรงผมต่างๆ จำนวน 2,600 รูป (เป็นรูปภาพขาว-ดำทั้งหมด)
- แยกหมวดหมู่ตามทรงผม โดยแยกออกเป็น 5 คลาส ได้แก่ twintail(ทวินเทล), ponytail(หางม้า), short(ผมสั้น), long(ผมยาว), braid(ผมเปีย)
- แต่ละคลาสประกอบด้วยรูปภาพ 500 - 550 รูป
- รูปภาพทั้งหมดถูกนำมาใช้ Train โมเดล CNN จำแนกทรงผมด้วยรูปภาพ

---
                
### โมเดลที่ใช้ Train

### Convolutional Neural Network (CNN)
\nConvolutional Neural Network (CNN) เป็นโครงข่ายประสาทเทียมที่ออกแบบมาเพื่อวิเคราะห์และจำแนกรูปภาพโดยเฉพาะ โครงสร้างหลักของ CNN มีองค์ประกอบสำคัญดังนี้:

- Convolution Layer → ดึงคุณลักษณะสำคัญของภาพ เช่น เส้น ขอบ หรือรูปทรง
- Pooling Layer → ลดขนาดข้อมูล แต่ยังคงคุณสมบัติสำคัญ ช่วยลดการคำนวณ
- Fully Connected Layer → เชื่อมโยงข้อมูลที่ผ่านการวิเคราะห์เพื่อทำการคาดการณ์
- Activation Function (ReLU, Softmax) → ช่วยให้โมเดลเรียนรู้ความซับซ้อนได้ดีขึ้น

CNN จึงเหมาะกับงานด้านการประมวลผลภาพ เช่น การจำแนกวัตถุ การตรวจจับใบหน้า และการแปลภาษาด้วยภาพ
                
---

### อธิบายโค้ด      

""")
    
    st.write("Import Library ที่จำเป็น:")
    st.image("NN/import.png", width=700)

    st.write("ดึงโฟลเดอร์รูปภาพจาก Google Drive:")
    st.image("NN/drive.png", width=700)

    st.write("ตั้งค่า Hyperparameters:")
    st.image("NN/set_parameters.png", width=700)

    st.write("รีสเกล และ split ข้อมูล:")
    st.image("NN/split.png", width=700)
    st.write("โหลดข้อมูล Train และ Validation")
    st.image("NN/load.png", caption="ใช้ค่า Mean ในการจัดการ", width=700)

    st.write("ตรวจสอบ Class ของทรงผม:")
    st.image("NN/check_class.png", width=700)

    st.write("Build โมเดล:")
    st.image("NN/build_model.png", width=700)

    st.write("Compile โมเดล:")
    st.image("NN/compile.png", width=700)

    st.write("Train โมเดล:")
    st.image("NN/train_model.png", width=700)
    st.image("NN/train_acc.png", caption="Train Result", width=700)

    st.write("แสดงผลลัพธ์ของการ Train:")
    st.image("NN/train_result.png", width=700)

    st.write("บันทึกโมเดล:")
    st.image("NN/save_model.png", width=700)

    st.markdown("""
                
---
                
## แนวทางในการพัฒนาโมเดล
- Dataset ควรมีจำนวนรูปภาพมากกว่านี้
- ควรมี Data Augmentation เพื่อลด Overfitting
- ออกแบบเลเยอร์ให้เหมาะสมกับข้อมูล
- ควรเพิ่มข้อมูลที่เป็นภาพสี(RGB) ใน Dataset
                
""")

#**************************************************************************************************************************************************#

# หน้าทดสอบพยากรณ์ Anime Hairstyle
elif page == "Anime Girl Hairstyle Prediction (Demo)":
    st.title("Anime Girl Hairstyle Prediction (Demo)")
    
    # อัปโหลดรูปภาพ
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพของคุณ", type=["jpg", "png", "jpeg"])

    # สุ่มรูปจาก dataset
    selected_image_path = None
    if image_paths:
        if st.button("🎲 สุ่มรูปภาพ"):
            selected_image_path = random.choice(image_paths)

    # แสดงรูปภาพที่อัปโหลดหรือสุ่มมา
    image = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", width=700, channels="BGR")

    elif selected_image_path:
        image = cv2.imread(selected_image_path, cv2.IMREAD_COLOR)
        st.image(image, caption=f"Random Image: {os.path.basename(selected_image_path)}", width=700, channels="BGR")

    # ถ้ามีรูป ให้ประมวลผลและทำนาย
    if image is not None:
        processed_image = preprocess_image(image)
        prediction = nn_model.predict(processed_image)
        predicted_label = np.argmax(prediction)

        if len(hairstyle_encoder.classes_) > predicted_label:
            predicted_hairstyle = hairstyle_encoder.inverse_transform([predicted_label])[0]
        else:
            predicted_hairstyle = "Unknown"

        confidence = np.max(prediction) * 100  

        # แสดงผลลัพธ์
        st.write("### 🤖 Model Prediction")
        st.write(f"**Hairstyle:** {predicted_hairstyle}")
        st.write(f"**Confidence:** {confidence:.2f}%")
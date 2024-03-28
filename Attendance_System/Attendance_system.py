
import tkinter
import numpy as np
import cv2
import os
import json
from sklearn.preprocessing import LabelEncoder
from tkinter import *
from PIL import Image
from tkinter import ttk



faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.createLBPHFaceRecognizer()

window = tkinter.Tk()
text_var = tkinter.StringVar()


def saving_data(data,x):
    file=open(x,"w",encoding='utf-8')
    json.dump(data,file,ensure_ascii=False)
    file.close()


def loading_data(x):
 
    file= open(x,"r",encoding='utf-8')
    data=json.load(file)
    file.close()
    return data
   

data = loading_data("data.txt")
data_attandance = loading_data("Attendance.txt")







def register():
    username = username_entry.get()
    
    if username != "" :
        if username not in data:
            count = 0
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)
            while True:
                ret, img = cam.read()
                img = cv2.flip(img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale( 
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(20, 20)) 
                count += 1
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    cv2.imwrite("dataset/user." + str(username_entry.get()) + "." + str(count) + ".jpg", gray[y:y + h, x:x + w])
                if count >= 200:
                    break
                cv2.imshow('video', img)
                cv2.waitKey(10)
            cam.release()
            cv2.destroyAllWindows()
            data[username]=username
            saving_data(data,"data.txt")
            text_var.set( f"Succesfully registered {username}")
        else:
            text_var.set("User name Already Exists...Try another one")
    else:
        text_var.set("Please provide UserName")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')
        id = os.path.split(imagePath)[-1].split(".")[1]

        faces = faceCascade.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

label_encoder = LabelEncoder()
encoded_labels_ids_reverse_new=dict()
def trainmodel(path):

    images,ids = getImagesAndLabels(path)
    global encoded_labels_ids_reverse_new 
    encoded_labels_ids = label_encoder.fit_transform(ids)
    encoded_labels_ids_reverse_old = dict(zip(encoded_labels_ids, ids))
    encoded_labels_ids_reverse_new=encoded_labels_ids_reverse_old

    recognizer.train(images,np.array(encoded_labels_ids))
    recognizer.write('trainer/trainer.yml')





def multi_users():
    username = username_entry.get()
    registed=[]
    notregist=[]
    definedd=[]
    indata= username.split(',')
    for i in indata:
      if i in data:
        registed.append(i)
      else:
        notregist.append(i)
    if len(registed)>0 :
       cam = cv2.VideoCapture(0)
       cam.set(3, 640)
       cam.set(4, 480)
       trainmodel("dataset")
       recognizer.read("trainer/trainer.yml")  
       face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

       while True:
          ret, frame = cam.read()
          img = cv2.flip(frame, 1)
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

          faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

          for (x, y, w, h) in faces:
              cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
              roi_gray = gray[y:y + h, x:x + w]
              id, confidence = recognizer.predict(roi_gray)
              id_name = encoded_labels_ids_reverse_new.get(id)
              if id_name in registed :
                   cv2.putText(img, str(id_name), (x + 18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   data_attandance[id_name]=id_name
                   saving_data(data_attandance,"Attendance.txt")
                   definedd.append(id_name)
                   
              else:
                cv2.putText(img, "Unknown", (x + 18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

          cv2.imshow("Prediction", img)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

       cam.release()
       cv2.destroyAllWindows()
    else:
           
            text_var.set("All names Not Registed Or Empty")
    unique_items = list(set(definedd))
    set1 = set(registed)
    set2 = set(unique_items)
    unique_elements_in_first_list = list[set1.difference(set2)]
    
    listss = {'Defined Well': unique_items, 'Not Registed': notregist, 'Registed And Not Define': unique_elements_in_first_list}
    def ef(x):
        root = tkinter.Tk()
        root.title("Details")
        for list_name, items in x.items():
             list_label = ttk.Label(root, text=f"{list_name} :", font=("Arial", 12, "bold"))
             list_label.pack(anchor="w", padx=10, pady=5)
             for item in items:
               item_label = ttk.Label(root, text=f"- {item}", font=("Arial", 10))
               item_label.pack(anchor="w", padx=20)
             

        
        root.mainloop()
    ef(listss)

def One_User():
    username = username_entry.get()
   
    if username != "" :
        if username in data:
            
                cam = cv2.VideoCapture(0)
                cam.set(3, 640)
                cam.set(4, 480)
               
                trainmodel("dataset")
                recognizer.read("trainer/trainer.yml")
                while True:
                    ret, img = cam.read()
                    img = cv2.flip(img, 1)
                    
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=5,
                        minSize=(20, 20))
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                        id_name = encoded_labels_ids_reverse_new.get(id)
                        if username == id_name:
                           cv2.putText(img, str(id_name), (x + 18, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                           cv2.imshow("cam", img)
                           cv2.waitKey(10)
                           data_attandance[username]=username
                           saving_data(data_attandance,"Attendance.txt")
                           text_var.set(f"Login successfully {id_name} ")
                           cam.release()
                           cv2.destroyAllWindows()
                           break
                        else:
                           cv2.imshow("cam",img)
                           cv2.waitKey(5)
                           text_var.set("UserName doesn't match person,Please Try Login Again ")
                           cam.release()
                           cv2.destroyAllWindows()
                           break
            
        else:
            text_var.set("UserName Not Registed ")
    else:
        text_var.set("Please provide username ")


def open_new_window():
    new_window = tkinter.Toplevel(window)
    new_window.title("Students List")
    new_window.geometry('800x500')
    
    
    student_names = []
    data_attandance = loading_data("Attendance.txt")
    for i in data_attandance:
        student_names.append(i)
    
    student_tree = ttk.Treeview(new_window, columns=("Index", "Name"), show="headings")
    student_tree.heading("Index", text="Index")
    student_tree.heading("Name", text="Name")
    for index, name in enumerate(student_names, start=1):
        student_tree.insert("", "end", values=(index, name))
    
    student_tree.pack(fill='both', expand=True)


def End_Session ():
    
    file_path = 'Attendance.txt'
    with open(file_path, 'w') as file:
      file.truncate()
      file.write('{}')
      file.close()
     





window.title("Attendance System")
window.geometry('900x550')
window.configure(bg='#fff')
window.resizable(False,False)
img= PhotoImage(file="1.png")
Label(window,image=img,bg='white',width=350,height=370).place(x=70,y=90)

frame=Frame(window,width=350,height=400,bg="white")
frame.place(x=480,y=70)
heading =Label(frame,text=' System Attendance ',fg='#57a1f8', bg='white',font=('Microsoft Yahei UI Light',24,"bold") )
heading.place(x=30,y=0)



username_label = tkinter.Label(
    frame, text="User Name",  fg='#57a1f8', bg='white',font=("Helvetica", 16)).place(x=30,y=60)

username_entry = tkinter.Entry(frame, font=("Arial", 16))
username_entry.place(x=30,y=100)


message_label = tkinter.Label(frame,textvariable=text_var,font=("Arial", 16),bg="lightgreen").place(x=30,y=145)

login_button = tkinter.Button(
    frame, text="Login",bg="#7b3aec", fg="#FFFFFF", width=9, height=1, font=("Arial", 16),command=One_User).place(x=30,y=230)

register_button = tkinter.Button(
    frame, text="New Student", bg="#7b3aec", fg="#FFFFFF", width=10, height=1,font=("Arial", 16),command=register).place(x=210,y=230)


Attendens_button = tkinter.Button(
    frame, text="Attendees", bg="#7b3aec", fg="#FFFFFF", width=9, height=1,font=("Arial", 16),command=open_new_window).place(x=30,y=310)


EndSession_button = tkinter.Button(
    frame, text="End Session", bg="#7b3aec", fg="#FFFFFF",width=10, height=1, font=("Arial", 16),command=End_Session).place(x=210,y=310)

multiuser_button = tkinter.Button(
    frame, text="Multi_User", bg="#7b3aec", fg="#FFFFFF",width=10, height=1, font=("Arial", 16),command=multi_users).place(x=110,y=360)


window.mainloop()


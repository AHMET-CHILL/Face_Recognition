import face_recognition
import cv2
import os

# Kişilerin resimlerinin bulunduğu klasör
images_folder = r"C:\Users\ahmet\Desktop\face_detactionv2\images"  # Bu klasörün içinde kişilerin resimleri olacak

# Yüz tanıma için kişi bilgilerini tutan dictionary
known_face_encodings = []
known_face_names = []

# Kişi resimlerini yükleyip yüz kodlarını çıkaran fonksiyon
def load_known_faces():
    for person_name in os.listdir(images_folder):
        person_folder = os.path.join(images_folder, person_name)
        if os.path.isdir(person_folder):
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                try:
                    # Resmi yükle ve yüz kodunu çıkar
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)
                    if face_encoding:
                        known_face_encodings.append(face_encoding[0])
                        known_face_names.append(person_name)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

# Kameradan görüntü almak için OpenCV
def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Yüzleri bul
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Yüzün kimliğini karşılaştır
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Eğer eşleşme bulunursa, ilk eşleşen kişiyi seç
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Yüzü çerçeveye al ve ismi yaz
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1) 

        # Sonuçları göster
        cv2.imshow('Video', frame)

        # 'q' tuşuna basıldığında çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kamera kaydını kapat
    video_capture.release()
    cv2.destroyAllWindows()

# Ana fonksiyon
def main():
    load_known_faces()  # Kişilerin yüzlerini yükle
    recognize_faces()   # Yüz tanımayı başlat

if __name__ == "__main__":
    main()

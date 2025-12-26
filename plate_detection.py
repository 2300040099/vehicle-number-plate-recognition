import cv2
import pytesseract

# ✅ Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'

# Load Haar cascade for Russian number plates
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Open video file or camera (replace "video.mp4" with 0 for webcam)
cap = cv2.VideoCapture("video.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plate_roi = frame[y:y + h, x:x + w]

        # Preprocess plate region
        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        plate_blur = cv2.bilateralFilter(plate_gray, 11, 17, 17)
        _, plate_thresh = cv2.threshold(plate_blur, 150, 255, cv2.THRESH_BINARY)

        # OCR to read plate text
        text = pytesseract.image_to_string(plate_thresh, config='--psm 8')
        text = text.strip()
        if text:
            print("Detected Number Plate:", text)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write frame to output video
    out.write(frame)
    cv2.imshow('Number Plate Recognition', frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

import cv2
import os
import numpy as np
import easyocr
from ultralytics import YOLO
from google import generativeai as genai
from PIL import Image

GOOGLE_API_KEY="Put your key here"
genai.configure(api_key=GOOGLE_API_KEY)

def filter_text(rgb_region,ocr_result,region_treshold):
    rectangle_size = rgb_region.shape[0] * rgb_region.shape[1]

    plate = []

    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1],result[0][0]))
        height = np.sum(np.subtract(result[0][2],result[0][1]))

        if length*height / rectangle_size > region_treshold:
            plate.append(result[1])

    
    return plate

def generate_text(img) -> str:
    # Load the model
    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        image = Image.open(img)
        # Provide image and prompt to extract text
        response = model.generate_content(
            [image,
            f'''
                You are an expert in number licence plate recognition:
                Return the number plate
            '''
            ]
        )

        return response.text
    
    except Exception as e:
        rgb_region = cv2.imread(img)

        # resizing the image
        resized_img = cv2.resize( 
        rgb_region, None, fx = 2, fy = 2,  
        interpolation = cv2.INTER_CUBIC)

        # grayscallin it
        image = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)

        
        reader = easyocr.Reader(['en'])
        ocr_results = reader.readtext(image)
        results = filter_text(image,ocr_results,region_treshold = 0.2)
        return ' '.join(results)


def inference(model_path):
    # Load a model
    model = YOLO(model_path)  # load a custom model

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not os.path.exists('detected'):
        os.makedirs('detected')

    count = 0
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(frame)

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the confidence and label
                conf = box.conf[0]
                label = model.names[int(box.cls[0])]

                region = frame[y1:y2, x1:x2]

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"./detected/img{count}.jpg", rgb_region)

                # Use Google's Gemini model for OCR
                license_plate = generate_text(f"./detected/img{count}.jpg")
                print(f"License Plate: {license_plate}")

                # Draw the label, confidence, and license plate on the frame
                cv2.putText(frame, f'{label} {conf:.2f} {license_plate}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                count = count + 1

        # Convert the frame back to BGR for displaying
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow('Webcam with Bounding Boxes', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Pause the loop for 2 seconds
        # time.sleep(2)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Perform inference using the webcam
inference('./model2.pt')

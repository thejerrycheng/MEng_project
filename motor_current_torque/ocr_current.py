import cv2
import pytesseract
import csv
import time

# If Tesseract is not automatically found by pytesseract, uncomment and set the path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_voltage_current(text):
    """
    Parses the OCR text to extract voltage (first line)
    and current (second line) assuming the text is:
    
    <Voltage>
    <Current>
    
    Returns (voltage, current) as floats if found, otherwise (None, None).
    """
    # Split lines
    lines = text.strip().splitlines()
    if len(lines) >= 2:
        try:
            voltage_str = lines[0].strip()
            current_str = lines[1].strip()
            
            # Attempt to parse them as floats
            voltage = float(voltage_str)
            current = float(current_str)
            return voltage, current
        except ValueError:
            return None, None
    else:
        return None, None

def main():
    # Path to your input video file
    video_path = "input_video.mp4"
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Open CSV file for writing
    with open('output.csv', mode='w', newline='') as csv_file:
        fieldnames = ['time_s', 'voltage', 'current']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        start_time = time.time()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # No more frames or cannot fetch frame
                break
            
            frame_count += 1
            
            # Optionally, crop the frame to just the region of interest (ROI) where text is found.
            # For example, ROI = frame[y:y+h, x:x+w]
            # Adjust these values as needed.
            # ROI = frame[100:200, 50:300]
            # For now, we assume the entire frame has the text.
            ROI = frame  
            
            # Convert ROI to grayscale
            gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            
            # (Optional) Threshold or other preprocessing to improve OCR accuracy
            # _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Apply OCR
            ocr_text = pytesseract.image_to_string(gray, lang='eng')
            
            # Extract voltage and current from OCR text
            voltage, current = extract_voltage_current(ocr_text)
            
            # Calculate elapsed time from start of the video read
            elapsed_time = time.time() - start_time
            
            if voltage is not None and current is not None:
                # Write to CSV
                writer.writerow({
                    'time_s': f"{elapsed_time:.2f}",
                    'voltage': f"{voltage:.3f}",
                    'current': f"{current:.3f}"
                })
            
            # If you want to visualize the frame with recognized text, uncomment:
            # cv2.imshow("Frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # Release the video capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

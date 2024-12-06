import torch
import cv2
import numpy as np
import time
import os
from pathlib import Path
from paddleocr import PaddleOCR

class NumberPlateDetector:
    def __init__(self, model_path, use_cuda=True):
        # Initialize YOLOv5 model
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model = self.model.to(self.device)
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(lang='en', use_gpu=use_cuda and torch.cuda.is_available(), use_angle_cls=True)
        
        # Text properties for visualization
        self.text_font = cv2.FONT_HERSHEY_PLAIN
        self.color = (0, 0, 255)
        self.text_font_scale = 1.25

    def extract_plate(self, image, bbox):
        """Extract the license plate region from the image."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        plate_img = image[y1:y2, x1:x2]
        return plate_img

    def perform_ocr(self, plate_img):
        """Perform OCR on the extracted plate image."""
        try:
            result = self.ocr.ocr(plate_img)
            if result and result[0]:  # Check if any text was detected
                text = result[0][0][1][0]  # Get the recognized text
                confidence = result[0][0][1][1]  # Get the confidence score
                return text, confidence
        except Exception as e:
            print(f"OCR Error: {str(e)}")
        return None, None

    def process_video(self, video_path, save_plates=False):
        """Process a single video file with ANPR and OCR."""
        frame = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_width = int(frame.get(3))
        frame_height = int(frame.get(4))
        size = (frame_width, frame_height)
        
        # Create output paths
        video_name = Path(video_path).stem
        output_path = f'output_{video_name}.mp4'
        if save_plates:
            plates_dir = f'plates_{video_name}'
            os.makedirs(plates_dir, exist_ok=True)
        
        # Initialize video writer
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        
        # Performance tracking
        prev_frame_time = 0
        frame_count = 0
        detected_plates = set()  # Track unique plates
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, image = frame.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Perform detection
            output = self.model(image)
            result = np.array(output.pandas().xyxy[0])
            
            # Process each detection
            for i in result:
                # Extract coordinates and confidence
                p1 = (int(i[0]), int(i[1]))
                p2 = (int(i[2]), int(i[3]))
                confidence = i[-3]
                if confidence > 0.75:
                    # Extract plate region and perform OCR
                    plate_img = self.extract_plate(image, i)
                    plate_text, ocr_confidence = self.perform_ocr(plate_img)
                    
                    if plate_text:
                        # Clean and standardize plate text
                        plate_text = ''.join(e for e in plate_text if e.isalnum())
                        
                        # Save plate image if option enabled
                        if save_plates and plate_text not in detected_plates:
                            plate_path = os.path.join(plates_dir, f'{plate_text}_{frame_count}.jpg')
                            cv2.imwrite(plate_path, plate_img)
                            detected_plates.add(plate_text)
                        
                        # Draw bounding box and text
                        cv2.rectangle(image, p1, p2, color=self.color, thickness=2)
                        text = f"{plate_text} ({ocr_confidence:.2f})"
                        cv2.putText(image, text, (p1[0], p1[1]-5),
                                self.text_font, self.text_font_scale,
                                self.color, 2)
            
            # Calculate and display FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time else 0
            prev_frame_time = new_frame_time
            cv2.putText(image, f"FPS: {int(fps)}", (7, 70), 
                       self.text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
            # Write and display frame
            # writer.write(image)
            cv2.imshow("ANPR System", image)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        frame.release()
        writer.release()
        print(f"Completed processing: {video_path}")
        print(f"Output saved as: {output_path}")
        print(f"Unique plates detected: {len(detected_plates)}")
        if detected_plates:
            print("Detected plates:", ', '.join(detected_plates))
        print()

def main():
    # Configuration
    model_path = r"best.pt"  # custom model path
    videos_dir = r"videos"    # directory containing input videos
    save_plates = True        # option to save individual plate images
    use_cuda = True          # use CUDA if available
    
    # Initialize detector
    detector = NumberPlateDetector(model_path, use_cuda)
    
    # Supported video formats
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Process all videos in directory
    for video_file in os.listdir(videos_dir):
        if video_file.lower().endswith(video_extensions):
            video_path = os.path.join(videos_dir, video_file)
            try:
                detector.process_video(video_path, save_plates)
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
    
    cv2.destroyAllWindows()
    print("Completed processing all videos.")

if __name__ == "__main__":
    main()
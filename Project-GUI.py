from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import object_detection as od
import imageio
import cv2
import numpy as np

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.line = []
        self.light_roi = []
        self.rect = []
        self.master.title("Traffic Violation")
        self.pack(fill=BOTH, expand=1)

        self.counter = 0
        self.mode = None

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        
        analyze = Menu(menu)
        analyze.add_command(label="Region of Interest (Stop Line)", command=self.regionOfInterest)
        analyze.add_command(label="Traffic Light ROI", command=self.trafficLightROI)
        menu.add_cascade(label="Analyze", menu=analyze)

        self.filename = "C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/home.jpg"
        self.imgSize = Image.open(self.filename)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)
        
        self.canvas = Canvas(master=root, width=self.w, height=self.h)
        self.canvas.create_image(20, 20, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def open_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])

        if self.filename:
            cap = cv2.VideoCapture(self.filename)
            if not cap.isOpened():
                print("Error: Could not open video")
                return

            ret, image = cap.read()
            if ret:
                cv2.imwrite('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/preview.jpg', image)
                self.show_image('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/preview.jpg')
            else:
                print("Error: Could not read first frame")
            cap.release()

    def show_image(self, frame):
        self.imgSize = Image.open(frame)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)

        self.canvas.destroy()

        self.canvas = Canvas(master=root, width=self.w, height=self.h)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def regionOfInterest(self):
        self.mode = 'line'
        root.config(cursor="plus") 
        self.canvas.bind("<Button-1>", self.imgClick) 

    def trafficLightROI(self):
        self.mode = 'light'
        root.config(cursor="plus") 
        self.canvas.bind("<Button-1>", self.imgClick) 

    def client_exit(self):
        exit()

    def imgClick(self, event):
        if self.counter < 2:
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            if self.mode == 'line':
                self.line.append((x, y))
            elif self.mode == 'light':
                self.light_roi.append((x, y))
            self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
            self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
            self.counter += 1

        if self.counter == 2:
            self.canvas.unbind("<Button-1>")
            root.config(cursor="arrow")
            self.counter = 0

            img = cv2.imread('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/preview.jpg')
            if self.mode == 'line' and len(self.line) == 2:
                print(f"Stop line: {self.line}")
                cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)
            elif self.mode == 'light' and len(self.light_roi) == 2:
                print(f"Traffic light ROI: {self.light_roi}")
                cv2.rectangle(img, self.light_roi[0], self.light_roi[1], (0, 0, 255), 3)
            
            cv2.imwrite('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/copy.jpg', img)
            self.show_image('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/copy.jpg')

            if self.mode == 'line':
                try:
                    self.main_process()
                    print("Executed Successfully!!!")
                    # Display final output preview with all annotations
                    self.show_image('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/output_preview.jpg')
                except Exception as e:
                    print(f"Error during video processing: {e}")

            self.line.clear()
            self.light_roi.clear()
            self.rect.clear()
            for i in self.pos:
                self.canvas.delete(i)

    def intersection(self, p, q, r, t):
        (x1, y1) = p
        (x2, y2) = q
        (x3, y3) = r
        (x4, y4) = t

        a1 = y1-y2
        b1 = x2-x1
        c1 = x1*y2-x2*y1

        a2 = y3-y4
        b2 = x4-x3
        c2 = x3*y4-x4*y3

        if(a1*b2-a2*b1 == 0):
            return False
        x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
        y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)

        if x1 > x2:
            tmp = x1
            x1 = x2
            x2 = tmp
        if y1 > y2:
            tmp = y1
            y1 = y2
            y2 = tmp
        if x3 > x4:
            tmp = x3
            x3 = x4
            x4 = tmp
        if y3 > y4:
            tmp = y3
            y3 = y4
            y4 = tmp

        if x >= x1 and x <= x2 and y >= y1 and y <= y2 and x >= x3 and x <= x4 and y >= y3 and y <= y4:
            return True
        return False

    def main_process(self):
        video_src = self.filename
        cap = cv2.VideoCapture(video_src)

        try:
            reader = imageio.get_reader(video_src)
            fps = reader.get_meta_data()['fps']
        except Exception as e:
            print(f"Error initializing video reader: {e}")
            cap.release()
            raise

        try:
            writer = imageio.get_writer('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Resources/output/output.mp4', fps=fps)
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            cap.release()
            reader.close()
            raise
            
        obj_thresh = 0.5
        nms_thresh = 0.45
        dcnt = 1
        last_frame = None

        while True:
            ret, image = cap.read()
           
            if not ret or image is None:
                writer.close()
                if last_frame is not None:
                    cv2.imwrite('C:/Users/n2007/Traffic-Signal-Violation-Detection-System/Images/output_preview.jpg', last_frame)
                break
            
            results = od.model.track(image, conf=obj_thresh, iou=nms_thresh, persist=True)
            boxes = results[0].boxes
            image2, dcnt = od.draw_boxes(image, boxes, self.line, od.labels, obj_thresh, dcnt, light_roi=self.light_roi if self.light_roi else None)
            
            image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            writer.append_data(image2_rgb)
            last_frame = image2  # Save the last processed frame with all annotations

            print(f"Frame {dcnt}")
            dcnt += 1

        cap.release()
        reader.close()

root = Tk()
app = Window(root)
root.geometry("%dx%d" % (535, 380))
root.title("Traffic Violation")
root.mainloop()
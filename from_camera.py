import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# matplotlib inline
import math
from moviepy.editor import VideoFileClip


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    blurredImage = gaussian_blur(image, 5)
    edgesImage = canny(blurredImage, 50, 100)
    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    regionInterestImage = region_of_interest(edgesImage, vertices)

    lineMarkedImage = hough_lines(regionInterestImage, 1, np.pi/180, 50, 100, 160)
    # draw output on top of original
    return weighted_img(lineMarkedImage, image)

# Deklarasi
kernel_size = 5
low_threshold = 50
high_threshold = 200
rho = 1 # Jarak resolusi grid Hough (dalam pixel) 
theta = np.pi/180 # resolusi angulara grid Hough (dalam radian) 
threshold = 50   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 1 #minimum number of pixels making up a line
max_line_gap = 250   # maximum gap in pixels between connectable line segments


cam = cv2.VideoCapture(0)
# cam.set(4,1280) # set Width
# cam.set(3,720) # set Height
if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    _, image = cam.read()
    if not _:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Fungsi-fungsi
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()  

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    # print (lines)
    # print (type(lines))
    # Iterate over the output "lines" and draw lines on a blank image
    if not type(lines) == type(None):
        for line in lines:
            print (line)
            for x1,y1,x2,y2 in line:
                # print (x1, y1, x2, y2)
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    else:
        print('Tidak terdeteksi garis! Ubah posisi kamera')
    

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)


    # Menampilkan hasil
    cv2.imshow('Camera', lines_edges)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()     


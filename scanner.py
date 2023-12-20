import cv2
import numpy as np
import pytesseract
from PIL import Image
import urllib.request as request


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
ocr_config = r"--psm 5 --oem 3"


def resize_img(img, scale):
    image = img
    # Get the original height and width of the image
    height, width = image.shape[:2]
    # Define the percentage by which you want to scale the image
    scale_percent = scale

    # Calculate the new dimensions
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)

    # Perform the resizing while maintaining the aspect ratio
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image


def rectify(h):
    h = h.reshape((4,2)) #Also add Exception Handling Here
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


url = 'http://10.7.84.34:8080/photo.jpg'
count = 0
while True:
    # load image here.
    img = request.urlopen(url)
    img_bytes = bytearray(img.read())
    img_np = np.array(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_np,-1)

    # convert to grayscale and blur to smooth
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 50)


    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page_contour = None

    # get approximate contour
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * p, True)

        if len(approx) == 4:
            page_contour = approx
            break


    # mapping page_contour points to 650x650 quadrilateral
    approx = rectify(page_contour)
    pnts2 = np.float32([[0,0],[650,0],[650,650],[0,650]])

    matrix = cv2.getPerspectiveTransform(approx,pnts2)
    warped_bev = cv2.warpPerspective(image,matrix,(800,800))

    cv2.drawContours(image, [page_contour], -1, (0, 255, 0), 2)
    warped_bev = cv2.cvtColor(warped_bev, cv2.COLOR_BGR2GRAY)

    x_start, y_start = 0, 0  # Starting coordinates
    x_end, y_end = 650, 650    # Ending coordinates
    # Crop the image using NumPy array slicing
    warped_bev = warped_bev[y_start:y_end, x_start:x_end]
    warped_bev = cv2.rotate(warped_bev, cv2.ROTATE_90_CLOCKWISE)
    th3 = cv2.adaptiveThreshold(
        src=warped_bev,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    cv2.imshow("Thresh gauss.jpg", th3)
    #cv2.imshow("warped_bev.jpg", warped_bev)
    cv2.imshow("Outline.jpg", image)
    if cv2.waitKey(1) == ord('s'):
        count += 1
        # binary_img_pil = Image.fromarray(th3)
        # binary_img_pil.save(f'binary_save_{count}.jpg')
        warped_img_pil = Image.fromarray(warped_bev)
        warped_img_pil.save(f'warped_save_{count}.jpg')
        text = pytesseract.image_to_string(th3)
        print(text)

    cv2.waitKey(1)









# other thresholding methods
"""
ret,th1 = cv2.threshold(warped_bev,127,255,cv2.THRESH_BINARY)
#ret, binarized = cv2.threshold(warped_bev,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh1 = cv2.threshold(warped_bev,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(warped_bev,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(warped_bev,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(warped_bev,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(warped_bev,127,255,cv2.THRESH_TOZERO_INV)

cv2.imshow("Thresh Binary", thresh1)
cv2.imshow("Thresh Binary_INV", thresh2)
cv2.imshow("Thresh Trunch", thresh3)
cv2.imshow("Thresh TOZERO", thresh4)
cv2.imshow("Thresh TOZERO_INV", thresh5)
cv2.imshow("Otsu's.jpg", binarized)
"""





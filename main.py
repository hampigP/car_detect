import cv2
import os
import numpy as np
import re
import pytesseract
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt

# 設置轉換圖片的資料夾

# 原始照片
input_folder = 'C:/car_detect/old'
# 轉換照片
output_folder = 'C:/car_detect/new'
# 有註解的照片
annotated_folder = 'C:/car_detect/annotated'
# 只有車牌的照片
cropped_folder = 'C:/car_detect/cropped'
# 車牌文字檔
output_text_file = 'C:/car_detect/car_plates.txt'
with open(output_text_file, 'w') as f:
    # 清空文件
    f.write('')

# 如果轉換圖片後的資料夾不存在，則創造它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(annotated_folder):
    os.makedirs(annotated_folder)

if not os.path.exists(cropped_folder):
    os.makedirs(cropped_folder)

# 分類器
detector = cv2.CascadeClassifier('haar_carplate.xml')
if detector.empty():
    print("Failed to load haar_carplate.xml")
    exit()

# 將原始圖片進行轉換並檢測車牌
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, filename)

        # 讀取圖片
        img = cv2.imread(img_path)
        if img is not None:
            # 調整照片大小
            resize_img = cv2.resize(img, (300,225))

            # 將調整過後的照片放入new資料夾中
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resize_img)
            print(f"Saved resized image to {output_path}")

            # 檢測車牌
            gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
            signs = detector.detectMultiScale(gray, minSize=(76,20), scaleFactor=1.1, minNeighbors=4)
            if len(signs) > 0:
                for(x, y, w, h) in signs:
                    cv2.rectangle(resize_img, (x,y), (x + w, y + h), (0, 0, 225), 2)
                print(f"Detect car plates in {filename}: {signs}")

                # 將檢測到車牌的照片存入annotate資料夾
                annotated_path = os.path.join(annotated_folder, filename)
                cv2.imwrite(annotated_path, resize_img)
                print(f"Saved annotated image to {annotated_path}")

                # 擷取車牌的照片存入cropped資料夾
                for (x, y, w, h) in signs:
                    image1 = Image.open(output_path)
                    # 擷取車牌
                    image2 = image1.crop((x, y, x + w, y + h))
                    # 將照片轉換尺寸為140*40
                    image3 = image2.resize((140,40), Image.LANCZOS)
                    # 將圖片轉為灰階
                    img_gray = np.array(image3.convert('L'))
                    _, img_thre = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
                    cropped_path = os.path.join(cropped_folder, filename)
                    cv2.imwrite(cropped_path, img_thre)
                    print(f"Saved cropped car plate to {cropped_path}")

                    

                    # 使用Tesseract OCR 將車牌號碼轉為文字
                    # text = pytesseract.image_to_string(img_thre, config='--psm 8')
                    # text = text.strip()
                    text = pytesseract.image_to_string(img_thre, config='--psm 10')
                    text = text.strip()
                    # res = re.search(r'[A-z]{2}-[0-9]{3}-[A-Z]{2}', text).group()
                    print(f"Detected text: {text}")

                    # 將車牌號碼寫入car_plates.txt
                    with open(output_text_file, 'a') as f:
                        f.write(f"{filename}: {text}\n")
            else:
                print(f"No car plates detected in {filename}")
        else:
            print(f"Saved resized image to {img_path}")
print('擷取車牌結束')



# # 讀取彩色車牌影像
# img = cv2.imread("RAT6232.jpg")
# plt.imshow(img)
# plt.show()

# # 將圖片更改為尺寸300*225
# img_new = cv2.resize(img, (300, 225))
# plt.imshow(img_new)
# plt.show()

# # 將圖像轉為灰階
# img_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
# # plt.imshow(img_gray)
# # plt.show()

# # 用濾波器模糊影像雜訊
# img_bilateral = cv2.bilateralFilter(img_gray, 15, 100, 100)
# # plt.imshow(img_bilateral)
# # plt.show()

# # 邊緣檢測
# img_canny = cv2.Canny(img_bilateral, 300, 500)
# # plt.imshow(img_canny)
# # plt.show()

# # 影像二值化
# thresh = 128
# maxval = 255
# ret, img_binary = cv2.threshold(img_canny, thresh, maxval, cv2.THRESH_BINARY)
# # plt.imshow(img_binary)
# # plt.show()

# # 閉操作運算
# kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (40,33))
# img_close = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernal)
# # plt.imshow(img_close)
# # plt.show()

# # 找出影像中車牌區域
# contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# result = None
# for i in contours:
#     x, y, width, hight = cv2.boundingRect(i)
#     if width > 2*hight:
#         # plt.imshow(img[y:(y+hight), x:(x+width)])
#         # plt.show()
#         result = img[y:(y+hight), x:(x+width)]

# # 辨識車牌上的英文和數字
# text = pytesseract.image_to_string(img, config='--psm 11')
# text = text.replace(" ", "")
# res = re.search(r'[A-z]{2}-[0-9]{3}-[A-Z]{2}', text).group()
# print(res)
import cv2
path =  "D:/NHAN_DIEN_HEO/NHANDIENHEO/label/"
for i in range(0, 13):  
    print(path+str(i)+'.jpg') 
    img = cv2.imread(path+'images' +str(i)+'.jpg') 
    img5050 = cv2.resize(img, (100,100))
    cv2.imshow("img",img5050)
    cv2.waitKey(20)
    cv2.imwrite('D:/NHAN_DIEN_HEO/NHANDIENHEO/label/'+'heo'+str(i)+'.jpg', img5050)
import os, cv2
import matplotlib.pyplot as plt
from tqdm import tqdm 

duowngddan = os.getcwd()
##duowngddan = r'/media/asrock/New Volume/VNPhatLoc/VuIbcCrawler/deeecaptttttttttttcha/crawl_image/mb_vietcom_bidv_agri/tessst'
##duowngddan = input('đường dẫn folder chứa dữ liệu: ')
##ss = [s for s in os.listdir(duowngddan) if (s.endswith('.png') and len(s) == len('873156.png'))]
##print(len(ss))

##import shutil
##for s in ss:
##    source = os.path.join(duowngddan, s)
##    destination = os.path.join(duowngddan, 'mb', s)
##    shutil.copyfile(source, destination)

##sokitu = input('số kí tự captcha: ')
##try:
##    sokitu = int(sokitu)
##except:
##    sokitu = len('873156.png')
##    print('số kí tự captcha vừa nhập không phải là số, mặc định sẽ là: ', sokitu)
labeled = 0
##print(len([s for s in os.listdir(duowngddan) if len(s)==sokitu]))
for s in tqdm(os.listdir(duowngddan)):
##    try:
        if not (s.endswith('.png') or s.endswith('.jpg')):#not s.startswith('346523'):#
            print("not (s.endswith('.png') or s.endswith('.jpg')): ", s)
            labeled+=1
            print(labeled)
            continue
        z = os.path.join(duowngddan, s)
        print('****', s)
        
##        if len(s)==sokitu:
####            xacnhan = input("nhãn đã được đánh giá trị, gõ ! để bỏ qua")
####            if xacnhan=='!':
##                continue
    
##        cv2.destroyAllWindows()
        zz = cv2.imread(z)
        cv2.imshow('', zz)
        cv2.waitKey(10)

##        from PIL import Image
##        with Image.open(z) as im:
##            im.show()
    
        ss = input('đánh nhãn: ').strip()
        if ss == '!' or ss=='':
##            os.rename(z, os.path.join(duowngddan, s[:-len('.jpg')]+'.jpg'))
            continue 
        sz = os.path.join(duowngddan, ss+'.jpg')
        print('####', sz)
        os.rename(z, sz)
##    except FileExistsError:
##        sz = os.path.join(duowngddan, ss+'(same).jpg')
##        s1z = input('FileExistsError: '+ sz+ ' ?')
##        if s1z.strip()=='ok':
##            os.rename(z, sz)
##

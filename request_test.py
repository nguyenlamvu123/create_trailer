import requests
import os 

urrrl = [
##    ('1712bidvsai1', 'http://cc.duongduhoc.com/5char/', ''),#sai 1/71
##    ('1712vietcom', 'http://cc.duongduhoc.com/5chaar/', ''),#sai 0/49
##    ('2812mbbank', "http://cc.duongduhoc.com/", ''),#sai 1/49 0/49 5/49
##    (
##        os.path.join(os.path.dirname(os.getcwd()), 'crawl_image', 'mb_vietcom_bidv_agri'),
##        "http://cc.duongduhoc.com/agri/",#"http://127.0.0.1:8000/agri/",#
##        'agri',
##        ),#sai 1/49 0/49 5/49#agri 
##    (
##        os.path.join(os.path.dirname(os.getcwd()), 'crawl_image', 'mb_vietcom_bidv_agri'),
##        'http://cc.duongduhoc.com/5char/',#'http://127.0.0.1:8000/5char/',#
##        'bidv',
##        ),#sai 1/71#bidv 
##    (
##        os.path.join(os.path.dirname(os.getcwd()), 'crawl_image', 'mb_vietcom_bidv_agri'),
##        'http://cc.duongduhoc.com/vietcom/',#'http://127.0.0.1:8000/vietcom/',#
##        'vcom',
##        ),#sai 0/49#vietcom 
##    (
##        os.path.join(os.path.dirname(os.getcwd()), 'crawl_image', 'mb_vietcom_bidv_agri'),
##        'http://cc.duongduhoc.com/',#'http://127.0.0.1:8000/',#
##        'mb',
##        ),#sai 0/49#mb  
    (
        os.path.join(os.path.dirname(os.getcwd()), 'crawl_image', 'mb_vietcom_bidv_agri'),
        'http://cc.duongduhoc.com/6char/',#'http://127.0.0.1:8000/6char/',#
        'vtin',
        ),#sai 0/49#mb  
    ]
##url = "http://cc.duongduhoc.com/"
##url = "http://127.0.0.1:8000/sbo/imaggge"

def samgiongzon(i):
    import base64
    image = open(i, 'rb')#open binary file in read mode
    image_read = image.read()
    image.close()
    ##image_64_encode = base64.b64encode(image_read)
    image_64_encode = base64.b64encode(image_read).decode('utf-8')
##    print(i);print(image_64_encode+ '\n')
    return image_64_encode

def tessst(
    bas_e64 = """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""",
    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'cache-control': "no-cache",
        'postman-token': "70b1d862-3d8f-3c3a-cbb1-d319fb421b52"
        },
    ):
    global url#, thutu 
    
    ##payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; "
    ##payload += "name=\"immmg\"; filename=\"6.jpg\"\r\n"
    ##payload += "Content-Type: image/jpeg\r\n\r\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    ##payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"basase64\"\r\n\r\n"+\
    ##          "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"+\
    ##          "\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"basase64\"\r\n\r\n"+\
              bas_e64+\
              "\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    response = requests.request("POST", url[1], data=payload, headers=headers)
##    print(response.text)
    return response.text+ '.png'

for url in urrrl:#for thutu, url in enumerate(urrrl[3:]):#if True:
    sammmgiongzon = url[0]#sammmgiongzon = url[thutu][0]
    ##sammmgiongzon = os.path.join(
    ##    os.path.dirname(
    ##        os.getcwd()
    ##        ),
    ##    'crawl_image',
    ##    url[2][0],#
    ##    '1',
    ##    )

    print(sammmgiongzon);print(url[2])#;print(url[thutu][2])
    import time, gc
    from tqdm import tqdm
    for i1 in tqdm(
        [i for i in os.listdir(sammmgiongzon) if i.endswith(url[2]+ '.jpg')][:30]#('mb.jpg')]
##        [i for i in os.listdir(sammmgiongzon) if i.endswith(url[thutu][2]+ '.jpg')]#('mb.jpg')]
        ):#[2:4]:#range(1,7):
        start = time.time()
        decapt = tessst(
                bas_e64 = samgiongzon(
                os.path.join(
                    sammmgiongzon,
                    i1,
                    )
                )
            )#;print(i1)
        os.rename(
            os.path.join(
                sammmgiongzon,
                i1,
                ),
            os.path.join(
                sammmgiongzon,
                decapt,
                ),
            )
        print(i1)#;time.sleep(3)
        print(time.time() - start)
        del decapt
        gc.collect()
        
    ##    input('enter để tiếp tục!')

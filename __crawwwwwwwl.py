##site:https://stackoverflow.com/questions/58170965/how-to-use-requests-library-with-selenium-in-pythonfrom selenium import webdriver
##site:https://stackoverflow.com/questions/48308973/python-download-image-from-https-aspx
from selenium import webdriver
import requests
from webdriver_manager.chrome import ChromeDriverManager

import time

options = webdriver.ChromeOptions()

options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-infobars")
options.add_argument("--disable-extensions")
options.add_argument("--disable-notifications")
##options.add_argument("--auto-open-devtools-for-tabs")
##options.add_argument('--headless')#not show window
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-using")
options.add_argument("--start-maximized")
prefs = {"profile.default_content_setting_values.notifications": 2}
options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)  #
##driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)  #
##driver.get('https://www.agbong88.com/')
##driver.get("https://www.vn138.com/")
##driver.get("https://www.12bet.com/vn/home")
driver.get('https://online.mbbank.com.vn/pl/login?returnUrl=%2F');time.sleep(2)
##driver.get('https://vcbdigibank.vietcombank.com.vn/login?returnUrl=%2F')
##driver.get('https://smartbanking.bidv.com.vn/dang-nhap')
##pull_image = driver.get('https://ibank.agribank.com.vn/ibank/HCIC/capimg.jsp')#('https://ibank.agribank.com.vn/ibank/index.jsp')

def agbonnnnnnnng88():
    global driver
    driver.find_elements_by_xpath(
        '//input[@placeholder="Username"]'
    ##    '//input[@id="txtUserName"]'
            )[0].send_keys('B8UI30103')
    driver.find_elements_by_xpath(
        '//input[@placeholder="Password"]'
    ##    '//input[@id="txtPassWord"]'
        )[0].send_keys('Qq168168@@@')
    from selenium.webdriver.common.keys import Keys
    driver.find_elements_by_xpath(
        '//input[@id="btnLogin"]'
        )[0].send_keys(Keys.RETURN)
##agbonnnnnnnng88()
def deeeeeeel(samgiongzon):
    try:
        del samgiongzon
    except:
        pass

for z in range(30):#if True:##
    try:
        try:
            test_image = driver.find_element_by_xpath(
                '//img[contains(@class, "ng-star")]'
                ).get_attribute(
                    'src'
                    )
        except:
            input('//img[contains(@class, "ng-star")] not found')
        samgiongzon = driver.find_element_by_xpath(
            "//mat-icon[@id='refresh-captcha']"
            )
        try:
            driver.execute_script("arguments[0].click();", samgiongzon)
        except:
            samgiongzon.click()
##    ##################################https://stackoverflow.com/questions/49627458/python-selenium-download-images-jpeg-png-or-pdf-using-chromedriver
        import urllib#.request
        from urllib.request import urlopen, Request
        urlopen(Request(test_image, headers={'User-Agent': 'Mozilla'}))
        urllib.request.urlretrieve(test_image, str(z)+ "mb.jpg")
##    ##################################
        driver.refresh();print(z)
        deeeeeeel(samgiongzon)
##        deeeeeeel(pull_image)
        deeeeeeel(test_image)
##        deeeeeeel(myfile)
##        deeeeeeel(s)
##        deeeeeeel(cookie)
##        deeeeeeel(selenium_user_agent)
        time.sleep(3)#(20)
    except Exception as e:
        print('errrrrrrrrrrrrrrrror', e);time.sleep(2)
##        agbonnnnnnnng88()
        try:
####        driver.get("https://www.vn138.com/")
            driver.get('https://online.mbbank.com.vn/pl/login?returnUrl=%2F')
        except:
            pass
        pass

for z in range(30):#if True:##
    try:
        with requests.get(
            'https://ibank.agribank.com.vn/ibank/HCIC/capimg.jsp',
            stream=True
            ) as pull_image:
##    https://stackoverflow.com/questions/57249863/what-is-difference-between-soup-of-selenium-and-requests

            with open(str(z)+ "agri.jpg", "wb") as myfile:#with open("samgiongzon.jpg", "wb") as myfile:#
                myfile.write(pull_image.content)
        with requests.get(
            'https://smartbanking.bidv.com.vn/w1/captcha/4d6e969f-02b7-f5c9-c450-76429813fd2c',
            stream=True
            ) as pull_image:
##    https://stackoverflow.com/questions/57249863/what-is-difference-between-soup-of-selenium-and-requests

            with open(str(z)+ "bidv.jpg", "wb") as myfile:#with open("samgiongzon.jpg", "wb") as myfile:#
                myfile.write(pull_image.content)
        with requests.get(
            'https://vcbdigibank.vietcombank.com.vn/w1/get-captcha/326f559e-4c54-d6ac-d28c-981b1fa02947',
            stream=True
            ) as pull_image:
##    https://stackoverflow.com/questions/57249863/what-is-difference-between-soup-of-selenium-and-requests

            with open(str(z)+ "vcom.jpg", "wb") as myfile:#with open("samgiongzon.jpg", "wb") as myfile:#
                myfile.write(pull_image.content)
        with requests.get(
            'https://api-ipay.vietinbank.vn/api/get-captcha/63qceqc69',
            stream=True
            ) as pull_image:
            #https://stackoverflow.com/questions/6589358/convert-svg-to-png-in-python
            #https://docs.wand-py.org/en/0.6.7/guide/install.html#install-imagemagick-on-windows
            import wand.image
            with wand.image.Image(
                blob=pull_image,
                format="svg",
                ) as image:
                png_image = image.make_blob("jpg")
##    https://stackoverflow.com/questions/57249863/what-is-difference-between-soup-of-selenium-and-requests
            with open(str(z)+ "vtin.jpg", "wb") as myfile:#with open("samgiongzon.jpg", "wb") as myfile:#
                myfile.write(png_image)#(pull_image.content)
##    ##################################https://stackoverflow.com/questions/49627458/python-selenium-download-images-jpeg-png-or-pdf-using-chromedriver
##        import urllib#.request
##        from urllib.request import urlopen, Request
##        urlopen(Request(z, headers={'User-Agent': 'Mozilla'}))
##        urllib.request.urlretrieve(z, "local-filename.jpg")
##    ##################################
        print(z)#driver.refresh();
##        deeeeeeel(samgiongzon)
        deeeeeeel(pull_image)
##        deeeeeeel(test_image)
##        deeeeeeel(s)
##        deeeeeeel(cookie)
##        deeeeeeel(selenium_user_agent)
        deeeeeeel(myfile)
        time.sleep(3)#(20)
    except Exception as e:
        print('errrrrrrrrrrrrrrrror', e);time.sleep(2)
##        agbonnnnnnnng88()
####        driver.get("https://www.vn138.com/")
        pass
deeeeeeel(z)
driver.quit()
deeeeeeel(driver)

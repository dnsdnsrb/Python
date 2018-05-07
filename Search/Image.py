import pyscreenshot as sc
import time

#if __name__ == "__main__": #pyscreenshot 쓸 때 필요, 이 파일(Image.py)가 직접 실행되면 이 부분을 실행하겠다는 의미
                           #Import될 경우 이 부분은 실행되지 않는다.
                           #pyscreenshot에 script가 포함되어 있어 이렇게 사용해야 하는 듯 보인다.
#    im = sc.grab()
#    im.show()
#    im.save("./image.jpg")


def Capture():
    im = [0, 0, 0, 0, 0]

    if __name__ == "__main__":
        for i in range(5):
            im[i] = sc.grab(bbox=(100,100,510,510)) #X1 Y1 X2 Y2, (X2 - X1, Y2 - Y1)
            time.sleep(0.5)

        for i in range(5):
            im[i].show()

Capture()
import pyautogui
import numpy as np
import tensorflow as tf
class Action():
    def __init__(self):
        self.size = pyautogui.size()

    def act(self, inputs):  #0 <= action <= 1
        select = np.argmax(inputs[0])
        # x = inputs[1][0] % self.size[0]
        x = inputs[1][0] % self.size[0]
        y = inputs[1][1] % self.size[1]

        z = int(self.size[1] - (inputs[1][2] % self.size[1]) )

        # if x == 0 and y == 0:   #(0,0)좌표를 넣으면 오류를 내서 꼼수로 해결
        #     x = 0.1

        print("act :", select, "x :", x, "y :", y, "z :", z)
        # Hscroll(s) ,Scroll(s), Move(x, y), Drag(x, y) * 3, Click(x, y) * 3, DoubleClick(x, y)
        if select == 0:
            print(x, y, z)
            pyautogui.hscroll(z, x=x, y=y)
        elif select == 1:
            print(x, y, z)
            pyautogui.scroll(z, x=x, y=y)
        elif select == 2:
            pyautogui.moveTo(x, y)
        elif select == 3:
            pyautogui.dragTo(x, y, button='left')
        elif select == 4:
            pyautogui.dragTo(x, y, button='right')
        elif select == 5:
            pyautogui.dragTo(x, y, button='middle')
        elif select == 6:
            pyautogui.click(x, y, button='left')
        elif select == 7:
            pyautogui.click(x, y, button='right')
        elif select == 8:
            pyautogui.click(x, y, button='middle')
        elif select == 9:
            pyautogui.doubleClick(x, y, button='left')

        print("done.")

        im = pyautogui.screenshot()
        im = im.transpose((0, 2, 1, 3))
        return im


if __name__ == '__main__':
    #Hscroll(s) ,Scroll(s), Move(x, y), Drag(x, y, b), Click(x, y, b, n)    11개
    x = np.random.rand(2,9)

    actor = Action()
    actor.act(x)
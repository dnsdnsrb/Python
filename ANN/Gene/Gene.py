import random
import pyautogui

class Link:
    def __init__(self):
        self.source = None
        self.target = None
        self.weight = 0

class Neuron:
    def __init__(self):
        self.link = []
        self.value = 0
        self.bias = 0
        self.decay = 1  #push방식에 필요

    def connect(self, target):
        new = Link()
        new.to = target
        # new.weight = 0.1

        self.link.append(new)

    #
    # def work(self):
    #     self.pull()
    #     if self.activation() == True:
    #         self.link[].target?source?.work()   #pull의 경우 source와 target 둘 다 필요 => link 구현이 곤란한 듯?

    def work(self):
        if self.activation() == True:
            self.push()


    def activation(self):


    def pull(self):
        self.link[].source

    def push(self):


class Life:
    def __init__(self):
        self.gene_informations = [] #변화 가능하게 할 모든 변수들, 이 목록에 있으면 합성, 돌연변이로 변할 수 있다.
        self.neurons = [Neuron() for _ in range(100)]

    def self_replication(self):
        # 돌연변이를 포함한 자가복제
        # 돌연변이 정도는 무작위
        pass

    # 뉴런에서 교차는 불가능하다고 판단 제외한다.
    # 따라서 자기발전이 필요하다.(학습이 필요하다는 의미)
    # def crossover(self, other):
    #     # 1. 동일 뉴런 개수일 때 허용할까?
    #     # 2. 초기 백업본을 가지고 있다가 동일한 형태?의 백업본을 가진 것과 진행
    #     # 3. 여러 백업본을 가진 상태에서 2를 진행
    #     # => 모든 방법의 전제조건 = 동일 뉴런 개수?
    #     # 합성할 뉴런들도 학습되도록?
    #     # => 합성할 뉴런들 목록이 동일한 경우 교차 진행?
    #     pass

    def mutation(self):
        # 무작위로 0~1 사이 수를 받아옴.
        # 항목 당 허용치(threshold)보다 높으면 작동
        # 항목 : 가중치에 임의의 값 더하기, 링크 변경(링크 번호에서 임의의 값 더하기로 변경)
        pass

    #기분이 기억, 학습에 영향을 미치는 것이 아닌가?
    #기분이 좋은 경우 => 좋다고 판단하여 학습되도록 만듦
    #기분이 안좋은 경우 => 안좋다고 판단하여 학습되지 않도록 함.
    #반면교사는? => 더 복잡한 형태의 기분이 필요

    def input(self):
        image = pyautogui.screenshot()
        #image[][][]? 전체에 대해 neurons[]에 흩뿌린다.
        #image[i][j][k] -> neurons[m] -> neuron[n]
        #방법은 for문으로 image 전체를 루프하면서, 무작위 뉴런에게 연결한다.

    def output(self):
        pass

    def save(self):
        #계속 진행을 위해 저장시킴.(확장자로 인식하게 할까?)
        pass

if __name__ == '__main__':
    a = Life()
    print(len(a.neurons))
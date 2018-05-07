import numpy as np
import random
import math
from numpy import linalg

from matplotlib import pyplot as plt
from openpyxl import load_workbook

file = load_workbook('line_estimation-data.xlsx')
data = file['Sheet1']
x_xlsx = data['E']
y_xlsx = data['F']


class Data:
    pass


data = Data()
data.x = []
data.y = []
data.index = []

# data.x = [0, 1]
# data.y = [1, 3]
# data.index = [0, 1]


for i in range(2, 226):
    #print(i)
    data.x.append(x_xlsx[i].value)
    data.y.append(y_xlsx[i].value)
    data.index.append(i - 2)


class Model:
    boundary = 5
    minnum = 150
    def __init__(self):
        self.gradient = 0.0
        self.height = 0.0
        self.inline = []
        self.error = 100
        self.error_variance = 100 #안 씀
        self.wrong = 1000
        self.data_variance = 0
        self.point1 = []
        self.point2 = []

x = 0.0
y = 0.0
#y = model.gradient * x + model.height


def linear_model(model, index):
    random_number1 = index[random.randint(0, len(index) - 1)]
    random_number2 = index[random.randint(0, len(index) - 1)]
    #print("coor", random_number1, random_number2)

    x1 = data.x[random_number1]
    x2 = data.x[random_number2]
    y1 = data.y[random_number1]
    y2 = data.y[random_number2]
    x_diff = x1 - x2
    y_diff = y1 - y2

    model.point1 = np.array([x1, y1])
    model.point2 = np.array([x2, y2])

    try:
        model.gradient = y_diff / x_diff
    except:
        linear_model(model, index)
        "re"
        return

    model.height = y1 - model.gradient * x1

def compute(model, boundary):
    inline = []
    errors = []
    model.error = 0
    x_values = []
    y_values = []

    model_p_sub = model.point2 - model.point1
    model_p_dist = linalg.norm(model.point2 - model.point1)

    for i in data.index:
        point = np.array([data.x[i], data.y[i]])
        distance = linalg.norm(np.cross(model_p_sub, model.point1 - point)) / model_p_dist

        #print(distance)
        # model.y = model.gradient * data.x[i] + model.height
        # model.x = (data.y[i] - model.height) / model.gradient
        # x_diff = data.x[i] - model.x
        # y_diff = data.y[i] - model.y
        # distance = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
        #print(i)
        model.error += abs(distance - boundary)
        errors.append(abs(distance - boundary))

        if distance <= boundary:
            inline.append(i)
            #model.error += abs(distance - boundary)
            #errors.append(abs(distance - boundary))
            x_values.append(data.x[i])
            y_values.append(data.y[i])

    try:
        #model.error /= len(inline)
        model.error /= len(data.x)
    except:
        print("divide by zero")
        pass
    #print("model error", model.error, len(errors))
    model.error_variance = np.var(errors)
    model.data_variance = np.var(x_values) + np.var(y_values)
    model.inline = inline

# elif model.error < model_best.error and model.error_variance < model_best.error_variance and len(model.inline) > Model.minnum:#
# elif model.error < model_best.error and len(model.inline) > Model.minnum:

def compare(model, model_best, bound_decay, op):
    #Model.boundary -= bound_decay
    #if len(model.inline) < Model.minnum:
    #    return model_best
    #bound를 통제할 방법이 필요하다, 현재는 그냥 막 줄어듦
    if op == 'var':
        if len(model.inline) > len(model_best.inline):# and model.error < model_best.error:
            #print("update")
            Model.boundary -= bound_decay
            return model
        elif model.error < model_best.error and model.data_variance > model_best.data_variance:# and model.error_variance < model_best.error_variance:
            #print("update2")
            Model.boundary -= bound_decay
            return model
        # elif model.variance < model_best.variance and len(model.inline) > model_best.minnum:
        #     print("update2")
        #     model.boundary -= 0.1
        #     return model
        else:
            return model_best

    elif op == 'normal':
        if len(model.inline) > len(model_best.inline):# and model.error < model_best.error:
            #print("update")
            Model.boundary -= bound_decay
            return model
        elif model.error < model_best.error and len(model.inline) > Model.minnum:
            #print("update2")
            Model.boundary -= bound_decay
            return model
        else:
            return model_best

    elif op == 'opti':
        if not len(model.inline) == len(model_best.inline):
            return model_best

        elif model.error < model_best.error:
            return model
        else:
            return model_best

    print("문제 발생")


def draw():
    plt.scatter(data.x, data.y, color='yellow')
    plt.scatter([data.x[i] for i in model_best.inline], [data.y[i] for i in model_best.inline], color='green')
    #plt.plot([])
    plt.show()

# linear_model([i for i in range(len(data.x))])
# compute(model_best.boundary)
# model_best = compare()

#op = 'var'

def run(model_best, op, iteration):
    for i in range(int(iteration)):
        if i % 100 == 0:
            print(i)
        model = Model()
        linear_model(model, model_best.inline)

        compute(model, Model.boundary)

        # print("model",", grad : ", model.gradient, "height : ",model.height,
        #       "error : ", model.error, "bound :", model.boundary,
        #       "num : ", len(model.inline), "var :", model.var)
        #
        # print("best",", grad : ", model_best.gradient, "height : ",model_best.height,
        #       "error : ", model_best.error, "bound :", model_best.boundary,
        #       "num : ", len(model_best.inline), "var :", model_best.var)

        model_best = compare(model, model_best, 0.0, op)
        #show_model(model_best)

    return model_best
    # print(i, ", grad : ", model_best.gradient, "height : ",model_best.height,
    #       "error : ", model_best.error, "bound :", model_best.boundary,
    #       "num : ", len(model_best.inline), "var :", model_best.variance)

def show_model(model):
    print("grad :", model.gradient, "height :",model.height,
          "error :", model.error, "bound :", model.boundary,
          "num :", len(model.inline), "var :", model.data_variance)

model_best = Model()
model_best.inline = data.index

iteration = 3000

Model.minnum = 100
Model.boundary = 0.5

user_bound = float(input("Inline bound를 입력하시오 : "))
user_minnum = input("최소 수 제한을 입력하시오 : ")
user_type = input("var 혹은 normal을 입력하시오 : ")

Model.minnum = user_minnum
Model.boundary = user_bound


model_best = run(model_best, user_type, iteration)
show_model(model_best)
model_best = run(model_best, 'opti', iteration / 3)
show_model(model_best)
draw()

#
# for i in range(iteration):
#     model = Model()
#     linear_model(model_best.inline, model)
#
#     compute(model_best.boundary)
#
#     # print("model",", grad : ", model.gradient, "height : ",model.height,
#     #       "error : ", model.error, "bound :", model.boundary,
#     #       "num : ", len(model.inline), "var :", model.var)
#     #
#     # print("best",", grad : ", model_best.gradient, "height : ",model_best.height,
#     #       "error : ", model_best.error, "bound :", model_best.boundary,
#     #       "num : ", len(model_best.inline), "var :", model_best.var)
#
#     model_best = compare(0.0, True)
#
#     # print(i, ", grad : ", model_best.gradient, "height : ",model_best.height,
#     #       "error : ", model_best.error, "bound :", model_best.boundary,
#     #       "num : ", len(model_best.inline), "var :", model_best.variance)
#
# print(", grad :", model_best.gradient, "height :",model_best.height,
#       "error :", model_best.error, "bound :", model_best.boundary,
#       "num :", len(model_best.inline), "var :", model_best.data_variance)

import random
import numpy as np
import input_data

class Question:
    def __init__(self):
        self.number = ["what is the number?",
                  "what's the number",
                  "what's the number?",
                  "can you read it?",
                  "can you read?",
                  "can you read"
                  "what's the image?",
                  "what is the image?",
                  "what are you seeing?",
                  "speak the number",
                  "speak the number.",
                  "recognize the image's number",
                  "recognize the number",
                  "recognize.",
                  "recognize",
                  "speak"]

        self.draw = ["draw.",
                "draw",
                "can you draw?",
                "draw the number",
                "write the number",
                "write",
                "write the image's number",
                "write the picture's number",
                "draw the image's number.",
                "can you write?",
                "just draw",
                "just write",
                "just write.",
                "draw what you saw",
                "draw what you see.",
                "draw what you saw."]

    def gen(self):
        question = random.choice(['n', 'b'])
        if question == 'n':
            question = random.choice(self.number)
        else:
            question = random.choice(self.draw)
        # print(question)
        return question

if __name__ == '__main__':
    q = Question()
    dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_label, test_data, test_label = dataset.train.images, dataset.train.labels, \
                                                     dataset.test.images, dataset.test.labels
    b = np.array([q.gen() for _ in range(5)])
    print(b)



    # a = np.empty((train_data.shape[0], train_data.shape[1], 1))
    # print(a.shape)
    # for i in range(len(train_data)):
    #     a[:, :, ]

    # np.stack((train_data, b))

    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # b = np.array([q.gen() for _ in range(9)])
    # b = np.reshape(b, [3,3])
    # c = np.stack((a, b), axis=1)
    # print(c)
    # print(c.shape)
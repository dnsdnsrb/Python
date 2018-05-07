import gzip
import os
import urllib.request
import numpy
import flags
SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"

def maybe_download(filename, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)

    filepath = os.path.join(work_directory, filename)

    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downoladed', filename, statinfo.st_size, 'bytes.')

    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')    #byte 순서, >는 빅 에디안을 의미
    return numpy.frombuffer(bytestream.read(4), dtype=dt)

def extract_images(filename):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream: #with as는 파일 오픈하고 자기에게 포함된 구문을 실행후 자동으로 닫는다.
        #데이터 제일 처음에 이렇게 되있어서 읽는듯?
        magic = _read32(bytestream) #MNIST image 정보가 맞는지 확인
        if magic != 2051:
            raise  ValueError(
                'Invalid magic number %d in MNIST image file : %s' %
                (magic, filename))
        num_images = _read32(bytestream)    #이미지 숫자 확인
        rows = _read32(bytestream)          #행렬 정보
        cols = _read32(bytestream)
        buf = bytestream.read(int(rows * cols * num_images)) #진짜 정보 읽기
        data = numpy.frombuffer(buf, dtype=numpy.uint8) #변형
        data = data.reshape(int(num_images), int(rows), int(cols), 1)  #배열로 변형?
        return data

def dense_to_one_hot(labels_dense, num_classes=10): #ex> 3이면, 0 0 1 0 0 0 0 0 0 0으로 변경
    num_labels = labels_dense.shape[0] #[a][b][c]이면 shape[0]이면 a를 반환한다.
    index_offset = numpy.arange(num_labels) * num_classes #arange(3) => [0,1,2], arange(3) * 10 => [0, 10, 20]
    labels_one_hot = numpy.zeros((num_labels, num_classes)) #0으로 채운 num_labels행, num_classes열을 새로 만든다.
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1    #flat[n] n번째에 1을 넣는다.
    return labels_one_hot

def extract_labels(filename, one_hot = False):  #extract_images랑 비슷
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        #데이터 읽기 전에 있는 정보들 읽음
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file : %s' %
                (magic, filename)
            )
        num_items = _read32(bytestream)
        buf = bytestream.read(int(num_items))    #진짜 데이터 읽기
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)   #데이터 형태 변형

        if one_hot:
            return  dense_to_one_hot(labels)

    return  labels

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:   #assert는 True이면 경고 찍는 용도
            assert images.shape[0] == labels.shape[0], (
                "images.shape : %s labels.shape: %s" % (images.shape, labels.shape)
            )

            self._num_examples = images.shape[0]

            assert images.shape[3] == 1 #왜 1이면 안되냐
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]) #3차원을 2차원으로 변경, 2, 3번째 배열이 합쳐짐.

            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property   #python은 private protected따위가 없다 => 대신 property를 이용한다.
    def images(self):   #객체에서 사용할 때는 classname.images라는 이름으로 사용됨, 내부에서 사용할 때는 self._image로 사용됨.
        return  self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return  self._num_examples

    @property
    def epochs_completed(self):
        return  self._epochs_completed

    def next_batch(self, batch_size, fake_data = False):    #사용되지 않은 함수
        """"""
        if fake_data:
            fake_image = [1.0 for _ in range(784)]  #'_'는 for(i = 0, i < 784, i++)에서 i를 의미한다고 보면된다.
            fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:

            self._epochs_completed += 1

            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir=flags.MNIST_DIR, fake_data=False, one_hot=False):
    class DatatSets(object):
        pass    #class 안을 안 만들고 통과, 단순히 DataSet 객체들을 묶어보내기 위해 사용하기 위해 만든 클래스이다.
    data_sets = DatatSets() #그리고 객체 생성
    if fake_data:
        data_sets.train = DataSet([], [], fake_data= True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    VALIDATION_SIZE = 5000

    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)

    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)

    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)

    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    #DataSet 객체를 DataSets객체로 묶어서 반환한다
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
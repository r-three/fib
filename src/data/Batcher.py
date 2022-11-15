from torch.utils import data
from src.data.Dataset import Dataset


class Batcher(object):
    '''
    Batcher is responsible for returning batches of data
    '''
    def __init__(self, datasetReader, createDataset_fn, train_batchSize, eval_batchSize):

        self.datasetReader = datasetReader
        self.createPytorchDataset_fn = createDataset_fn

        self.train_batchSize = train_batchSize
        self.eval_batchSize = eval_batchSize

        self.trainLoader = None
        self.devLoader = None
        self.testLoader = None
        self.mulChoiceLoader = None

    def _init_trainLoader(self):
        trainData = self.datasetReader.read_origData("train")
        train_pytorchDatasetClass = self.createPytorchDataset_fn(trainData)
        self.trainLoader = data.DataLoader(train_pytorchDatasetClass,
                                            batch_size=self.train_batchSize,
                                            shuffle=True,
                                            collate_fn=train_pytorchDatasetClass.collate_fn)

    def _init_devLoader(self):
        devData = self.datasetReader.read_origData("dev")
        dev_pytorchDatasetClass = self.createPytorchDataset_fn(devData)
        self.devLoader = data.DataLoader(dev_pytorchDatasetClass,
                                          batch_size=self.eval_batchSize,
                                          shuffle=False,
                                          collate_fn=dev_pytorchDatasetClass.collate_fn)

    def _init_testLoader(self):
        testData = self.datasetReader.read_origData("test")
        test_pytorchDatasetClass = self.createPytorchDataset_fn(testData)
        self.testLoader = data.DataLoader(test_pytorchDatasetClass,
                                          batch_size=self.eval_batchSize,
                                          shuffle=False,
                                          collate_fn=test_pytorchDatasetClass.collate_fn)

    def _init_mulChoiceLoader(self, mulChoiceFilepath):
        mulChoiceData = self.datasetReader.read_mulChoiceData(mulChoiceFilepath)
        mulChoice_pytorchDatasetClass = self.createPytorchDataset_fn(mulChoiceData)
        self.mulChoiceLoader = data.DataLoader(mulChoice_pytorchDatasetClass,
                                        batch_size=self.eval_batchSize,
                                        shuffle=False,
                                        collate_fn=mulChoice_pytorchDatasetClass.collate_fn)

    def get_trainBatches(self):
        if self.trainLoader is None:
            self._init_trainLoader()

        while True:
            for x in self.trainLoader:
                yield x

    def get_devBatches(self):
        if self.devLoader is None:
            self._init_devLoader()

        for x in self.devLoader:
            yield x

    def get_testBatches(self):
        if self.testLoader is None:
            self._init_testLoader()

        for x in self.testLoader:
            yield x

    def get_mulChoiceBatches(self, mulChoiceFilepath):
        if self.mulChoiceLoader is None:
            self._init_mulChoiceLoader(mulChoiceFilepath)

        for x in self.mulChoiceLoader:
            yield x

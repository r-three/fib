import json

class PredictionLogger(object):
    def __init__(self, logger_fp):
        self.logger_fp = logger_fp
        self.logger_file = open(self.logger_fp, 'w+')

    # From https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    def _convert_dictOfLists_to_listOfDicts(self, dictOfLists):
        listOfDicts = []
        for datapoint_values in zip(*dictOfLists.values()):
            listOfDicts.append(dict(zip(dictOfLists, datapoint_values)))
        return listOfDicts

    def log_batch(self, batchOf_evalInfo):
        listOf_evalInfo = self._convert_dictOfLists_to_listOfDicts(batchOf_evalInfo)
        for eval_info in listOf_evalInfo:
            self.logger_file.write(json.dumps(eval_info) + '\n')
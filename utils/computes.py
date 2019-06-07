import csv
import numpy as np
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def f1_score(outputs, targets, compute=1, delta=1e-11):
    true_sum = np.sum(targets)
    pred_sum = np.sum(outputs)
    tp = np.sum(np.bitwise_and(outputs, targets))
    fp = pred_sum - tp
    fn = true_sum - tp
    tn = len(targets) - true_sum - fp

    # if concerned label is 0 instead of 1
    if not compute:
        tn, tp = tp, tn
        fn, fp = fp, fn

    #define the precision recall and F1-score
    precision = float(tp)/(tp + fp + delta)
    recall = float(tp)/(tp + fn + delta) # is also sensitivity
    specificity = float(tn) / ( tn + fp + delta) # if tn + fp = tp + fn, then spec = 2* acc - precision
    F1 = 2.0*precision*recall/(precision+recall+delta) 
    return precision, recall, F1, specificity


def create_logger(prefix):
    log_file = 'logs/{}{}.log'.format(prefix, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(log_file), format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

# ----------------------------------------------------------------
# deserted -------------------------------------------------------
# ----------------------------------------------------------------
class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

    def info(self, printout):
        print(printout)
        self.log_file.write(printout+'\n')
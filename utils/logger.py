import os
import time


class Logger():
    def __init__(self, opt):
        self.log_file = open(os.path.join(opt.log_dir, 'log.txt'), 'a')
        now = time.strftime('%c')
        self.log_file.write('\n'+'='*6 + '%s' % now +'='*6+'\n')
        self.log_file.flush()

    def print_current_losses(self, epoch, iter, losses, t_comp, t_data):
        message = '\u25ba [Epoch %d-iter %d](compute/load time:%.4f/%4.f) ' % \
                  (epoch, iter, t_comp, t_data)

        for k, v in losses.items():
            kk = k.split('/')[-1]
            message += '%s: %.4f ' % (kk, v)
        print(message, flush=True)
        self.log_file.write('%s\n' % message)
        self.log_file.flush()

    def print_current_metrics(self, epoch, iter, metrics):
        message = '\u25bb metric [Epoch %d-iter %d] ' % \
                  (epoch, iter)
        for k, v in metrics.items():
            kk = k.split('/')[-1]
            message += '%s: %.4f ' % (kk, v)
        print(message, flush=True)
        self.log_file.write('%s\n' % message)

    def print_info(self, message):
        print(message, flush=True)
        self.log_file.write('\u2755 ' + message + '\n')



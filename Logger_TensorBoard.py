import tensorboard_logger.tensorboard_logger as Logger
import time
import os

class Logger_TensorBoard(Logger.Logger):
    def __init__(self, logdir="./logdir", flush_secs=2):
        super(Logger_TensorBoard, self).__init__(logdir=logdir, flush_secs=2)
        model_save_dir = f'{"./logdir"}/log_{time.strftime("%Y%m%d%H%M")}'
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

    def write(self, train_loss, train_acc, val_loss, val_acc, leraning_rate,epoch):
        self.log_value('train_loss', train_loss, step=epoch)
        self.log_value('train_acc', train_acc, step=epoch)
        self.log_value('val_loss', val_loss, step=epoch)
        self.log_value('val_acc', val_acc, step=epoch)
        self.log_value('leraning_rate', leraning_rate, step=epoch)

    def write_train(self, val_loss, val_acc, epoch):
        self.log_value('val_loss', val_loss, step=epoch)
        self.log_value('val_acc', val_acc, step=epoch)

    def write_loss(self, train_loss, train_acc, epoch):
        self.log_value('train_loss', train_loss, step=epoch)
        self.log_value('train_acc', train_acc, step=epoch)

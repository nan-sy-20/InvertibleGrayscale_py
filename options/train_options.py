import os
from .base_options import BaseOptions

FOLDER = os.path.basename(os.getcwd())
model_id = int(FOLDER[FOLDER.find('_')+1:])
data_id_dict = {'VOC2012':3000, 'EPFL':3001, 'VCIP':3002}


class TrainOptions(BaseOptions):
    def __init__(self, task_name, isTrain=True):
        super().__init__()
        self.isTrain = isTrain
        self.task_name = task_name[task_name.find('_')+1:] if '_' in task_name else task_name
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train')

        # log parameters
        parser.add_argument('--log_dir', type=str, default='logs/'+self.task_name+'/'+os.path.basename(parser.parse_known_args()[0].data_dir),
                            help='training logs are saved here')
        parser.add_argument('--eval_freq', type=int, default=500)
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=50,
                            help='frequency of evaluating and save the latest model')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epoch')
        parser.add_argument('--epoch_base', type=int, default=1,
                            help='the epoch base of the training (used for resuming)')
        parser.add_argument('--iter_base', type=int, default=1,
                            help='the iteration base of the training (used for resuming)')

        # model parameters
        parser.add_argument('--model', type=str, default=self.task_name,
                            help='choose which decolornet to use')


        # training parameters
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--nepochs', type=int, default=100,
                            help='number of epochs with the initial learning rate')
        parser.add_argument('--nepochs_decay', type=int, default=100,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--betas', type=float, default=(0.5, 0.99), help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--train_batch_size', type=int, default=8, help='the evaluation batch size')
        parser.add_argument('--eval_batch_size', type=int, default=6, help='the evaluation batch size')

        # display
        parser.add_argument('--display_id', type=int, default=100,
                            help='display_id > 0, use visdom')
        parser.add_argument('--no_html', action='store_true')
        parser.add_argument('--display_port', type=int, default=data_id_dict[os.path.basename(parser.parse_known_args()[0].data_dir)])
        parser.add_argument('--display_ncols', type=int, default=5)
        parser.add_argument('--display_server', type=str, default='http://localhost')
        parser.add_argument('--display_env', type=str, default=self.task_name)
        parser.add_argument('--display_freq', type=int, default=400)
        parser.add_argument('--update_html_freq', type=int, default=400)

        parser.isTrain = True
        parser.model = self.task_name
        return parser
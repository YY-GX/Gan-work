import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.models as models
import torch.nn as nn
from fastai.layers import MSELossFlat
# from fastai.vision import *
from tensorboardX import SummaryWriter
from torchvision.utils import *
import torchvision.transforms as transforms
import numpy as np



class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
 
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)
        self.writer = SummaryWriter(opt.log_dir) 
        
        #         YYpart
        self.netRgr = models.__dict__[self.opt.arch]().to(self.device)
        ly = nn.Sequential(
            nn.Dropout(p=opt.dropout),
            nn.Linear(opt.fc_input, 1)
        )   
        # self.netRgr.fc = nn.Linear(512, 1).to(self.device)
        self.netRgr.fc = ly.to(self.device)
        self.netRgr = nn.DataParallel(self.netRgr, device_ids=self.opt.gpu_ids)
        self.criterion = MSELossFlat()#.to(self.device)
        
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'Rgr']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'Rgr']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
#             self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netRgr.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # YY part
            # Give regression & G diff lr
            self.optimizer_G = torch.optim.Adam([
                {'params': self.netG_A.parameters()},
                {'params': self.netG_B.parameters()},
                {'params': self.netRgr.parameters(), 'lr': self.opt.rgr_lr, 'weight_decay': self.opt.wd_rgr},
            ], lr=opt.lr, betas=(opt.beta1, 0.999))
            print('optimizer_G param:')
            print(self.optimizer_G)
            
            
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.A_label = input['A_label' if AtoB else 'B_label'].to(self.device)
        self.B_label = input['B_label' if AtoB else 'A_label'].to(self.device)
#         print('0===================================')
#         print(self.real_B.size())
#         print('0===================================')


    def call_tensorboard(self):
        img_grid_fake = make_grid(self.fake_B, normalize=False, scale_each=True, nrow=4)  # normalize进行归一化处理
        self.writer.add_image('fake_CT', img_grid_fake, global_step=0)
        img_grid_real = make_grid(self.real_B, normalize=False, scale_each=True, nrow=4)  # normalize进行归一化处理
        self.writer.add_image('real_CT', img_grid_real, global_step=0)
        for i, (name, param) in enumerate(self.netRgr.named_parameters()):
            if 'bn' not in name:
                self.writer.add_histogram(name, param, 0)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        
#       YY part
#         instanceNorm = nn.BatchNorm2d(self.opt.batch_size, affine=False).to(self.device)
#         self.input_rgr = self.fake_B.transpose(0,1).contiguous()
#         self.input_rgr = instanceNorm(self.input_rgr)
#         self.input_rgr = self.input_rgr.transpose(0,1).contiguous()
#         print('-------')
#         print(self.fake_B.shape)
#         print('-------')
        if not (self.opt.change_lr == -1):
            self.optimizers[0].param_groups[2]['lr'] = self.opt.change_lr
        if not (self.opt.change_wd == -1):
            self.optimizers[0].param_groups[2]['weight_decay'] = self.opt.change_wd


        # normalize fake_CT
        instanceNorm = nn.BatchNorm2d(3).to(self.device)
        #instanceNorm = nn.DataParallel(instanceNorm, device_ids=self.opt.gpu_ids)
        self.input_rgr = instanceNorm(self.fake_B)
        
        # tensorboard
        self.call_tensorboard()
        
        # concate fake_CT and real_CT images and their labels. Do shuffling to both of them without damaging their mapping.       
        self.temp1 = torch.cat((self.input_rgr, instanceNorm(self.real_B)), 0).detach().cpu().numpy()
        state = np.random.get_state()
        np.random.shuffle(self.temp1)
        self.concat_tensor = torch.from_numpy(self.temp1).to(self.device)
        # self.concat_tensor = nn.DataParallel(self.concat_tensor, device_ids=self.opt.gpu_ids)
        
        self.temp2 = torch.cat((self.A_label, self.B_label), 0).detach().cpu().numpy()
        np.random.set_state(state)
        np.random.shuffle(self.temp2)
        self.concat_label = torch.from_numpy(self.temp2).to(self.device)
        # self.concat_label = nn.DataParallel(self.concat_label, device_ids=self.opt.gpu_ids)
        
        self.rgr_output = self.netRgr(self.concat_tensor).squeeze()
        


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # YYGX
#         lambda_rgr = self.opt.lambda_Rgr  
        lambda_rgr = self.lambda_rgr
        self.iseesee = lambda_rgr
        self.loss_rgr = self.criterion(self.rgr_output, self.concat_label) 
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_rgr * lambda_rgr

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        
#         YY part
    def loss_train(self):
        return self.loss_rgr, self.rgr_output, self.concat_label, self.loss_rgr * self.lambda_rgr, self.loss_G_A, self.loss_cycle_A, self.iseesee
    
    def loss_test(self, data, label):
        data = data.to(self.device)
        label = label.to(self.device)
        temp = self.netRgr(data)
        output = self.criterion(temp, label)
        return output

    def set_lambda_rgr(self, lambda_rgr):
        self.lambda_rgr = lambda_rgr
        
        
'''
=========================================================================================================
My Note
=========================================================================================================
I. Cyclegan calculation flow:
1. Pre
    1. set_input(): Get real imgs  (call in train.py)
    2. forward(): Get fake imgs
2. Train G
    1. set_requires_grad(): Fix D 
    2. backward_G() & optimizer_G.step(): Calc gradient of G & update the gradient
3. Train D
    1. set_requires_grad(): Unfix D 
    2. backward_D_A/B() & optimizer_D.step(): Calc gradient of D & update the gradient

II. Identity loss: For photo2painting
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#what-is-the-identity-loss-322-373-362
'''

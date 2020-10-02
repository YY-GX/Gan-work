import torch.nn as nn
import torchvision.models as models
import torch

# class MultiTaskModel(nn.Module):

#     def __init__(self):
#         super(MultiTaskModel, self).__init__()
#         self.model = models.resnet50(num_classes=1, pretrained=False)
        
#         self.share = nn.Sequential(*(list(self.model.children())[:-3]))
#         self.mr = nn.Sequential(*(list(self.model.children())[-3:-1]))
#         self.ct = nn.Sequential(*(list(self.model.children())[-3:-1]))
#         self.mr_lr = nn.Linear(2048, 1)
#         self.ct_lr = nn.Linear(2048, 1)
        
# #         self.mr = nn.Sequential()
# #         self.ct = nn.Sequential()
        
# #         self.mr.add_module('mr_resnet', *(list(model.children())[-3:-2]))
# # #         self.mr.add_module('mr_pool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
# #         self.mr.add_module('mr_lr', nn.Linear(4096, 1))
        
# #         self.ct.add_module('ct_resnet', *(list(model.children())[-3:-2]))
# # #         self.ct.add_module('ct_pool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
# #         self.ct.add_module('ct_lr', nn.Linear(4096, 1))

#     def forward(self, input_data_mr, input_data_ct):
#         if input_data_mr == None: # For validation or test
#             shared_data_ct = self.share(input_data_ct)
            
#             ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
            
#             return ct_output
        
        
#         shared_data_mr = self.share(input_data_mr)    
        
#         mr_output = self.mr_lr(self.mr(shared_data_mr).view(shared_data_mr.size()[0], -1))
        
#         shared_data_ct = self.share(input_data_ct)

#         ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
        
#         return mr_output, ct_output
    
checkpoint = torch.load('./checkpoints/model_best_pure.pth.tar')


class MultiTaskModel(nn.Module):

    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.model = models.resnet50(num_classes=1, pretrained=False)
        
        
        self.share = nn.Sequential(*(list(self.model.children())[:-5]))
        self.mr = nn.Sequential(*(list(self.model.children())[-5:-1]))
        self.ct = nn.Sequential(*(list(self.model.children())[-5:-1]))
        self.mr_lr = nn.Linear(2048, 1)
        self.ct_lr = nn.Linear(2048, 1)
        
    def forward(self, input_data_mr, input_data_ct):
        if type(input_data_mr).__name__ != 'Tensor': # For validation or test
            shared_data_ct = self.share(input_data_ct)
            
            ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
            
            return ct_output
        
        
        shared_data_mr = self.share(input_data_mr)    
        
        mr_output = self.mr_lr(self.mr(shared_data_mr).view(shared_data_mr.size()[0], -1))
        
        shared_data_ct = self.share(input_data_ct)

        ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
        
        return mr_output, ct_output
    
class MultiTaskModel1(nn.Module):

    def __init__(self):
        super(MultiTaskModel1, self).__init__()
        self.model = models.resnet50(num_classes=1, pretrained=False)
        
        self.share = nn.Sequential(*(list(self.model.children())[:-4]))
        self.mr = nn.Sequential(*(list(self.model.children())[-4:-1]))
        self.ct = nn.Sequential(*(list(self.model.children())[-4:-1]))
        self.mr_lr = nn.Linear(2048, 1)
        self.ct_lr = nn.Linear(2048, 1)
        
    def forward(self, input_data_mr, input_data_ct):
        if type(input_data_mr).__name__ != 'Tensor': # For validation or test
            shared_data_ct = self.share(input_data_ct)
            
            ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
            
            return ct_output
        
        
        shared_data_mr = self.share(input_data_mr)    
        
        mr_output = self.mr_lr(self.mr(shared_data_mr).view(shared_data_mr.size()[0], -1))
        
        shared_data_ct = self.share(input_data_ct)

        ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
        
        return mr_output, ct_output

class MultiTaskModel2(nn.Module):

    def __init__(self):
        super(MultiTaskModel2, self).__init__()
        self.model = models.resnet50(num_classes=1, pretrained=False)
        
        self.share = nn.Sequential(*(list(self.model.children())[:-3]))
        self.mr = nn.Sequential(*(list(self.model.children())[-3:-1]))
        self.ct = nn.Sequential(*(list(self.model.children())[-3:-1]))
        self.mr_lr = nn.Linear(2048, 1)
        self.ct_lr = nn.Linear(2048, 1)
        
    def forward(self, input_data_mr, input_data_ct):
        if type(input_data_mr).__name__ != 'Tensor': # For validation or test
            shared_data_ct = self.share(input_data_ct)
            
            ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
            
            return ct_output
        
        
        shared_data_mr = self.share(input_data_mr)    
        
        mr_output = self.mr_lr(self.mr(shared_data_mr).view(shared_data_mr.size()[0], -1))
        
        shared_data_ct = self.share(input_data_ct)

        ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
        
        return mr_output, ct_output
    

class MultiTaskModel3(nn.Module):

    def __init__(self):
        super(MultiTaskModel3, self).__init__()
        self.model = models.resnet50(num_classes=1, pretrained=False)
        self.model.load_state_dict(checkpoint, strict=False)
        
        self.share = nn.Sequential(*(list(self.model.children())[:-4]))
        self.mr = nn.Sequential(*(list(self.model.children())[-4:-1]))
        self.ct = nn.Sequential(*(list(self.model.children())[-4:-1]))
        self.mr_lr = nn.Linear(2048, 1)
        self.ct_lr = nn.Linear(2048, 1)
        
    def forward(self, input_data_mr, input_data_ct):
        if type(input_data_mr).__name__ != 'Tensor': # For validation or test
            shared_data_ct = self.share(input_data_ct)
            
            ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
            
            return ct_output
        
        
        shared_data_mr = self.share(input_data_mr)    
        
        mr_output = self.mr_lr(self.mr(shared_data_mr).view(shared_data_mr.size()[0], -1))
        
        shared_data_ct = self.share(input_data_ct)

        ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
        
        return mr_output, ct_output
    
class MultiTaskModel4(nn.Module):

    def __init__(self):
        super(MultiTaskModel4, self).__init__()
        self.model = models.resnet50(num_classes=1, pretrained=False)
        self.model.load_state_dict(checkpoint, strict=False)
        
        self.share = nn.Sequential(*(list(self.model.children())[:-3]))
        self.mr = nn.Sequential(*(list(self.model.children())[-3:-1]))
        self.ct = nn.Sequential(*(list(self.model.children())[-3:-1]))
        self.mr_lr = nn.Linear(2048, 1)
        self.ct_lr = nn.Linear(2048, 1)
        
    def forward(self, input_data_mr, input_data_ct):
        if type(input_data_mr).__name__ != 'Tensor': # For validation or test
            shared_data_ct = self.share(input_data_ct)
            
            ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
            
            return ct_output
        
        
        shared_data_mr = self.share(input_data_mr)    
        
        mr_output = self.mr_lr(self.mr(shared_data_mr).view(shared_data_mr.size()[0], -1))
        
        shared_data_ct = self.share(input_data_ct)

        ct_output = self.ct_lr(self.ct(shared_data_ct).view(shared_data_ct.size()[0], -1))
        
        return mr_output, ct_output
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.functional import normalize
import glob
import random
from PIL import Image
import torchvision.transforms as transforms




def activate_fn(x,inplace=True):
    return F.relu(x,inplace=inplace)
# base conv in paper
class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride,activate=None):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.activate = activate_fn if activate is None else activate
        
    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return self.activate(x)
    



def calc_pdf_cdf(x):

    zeros = torch.zeros_like(x).to(x)
    ones  = torch.ones_like(x).to(x)
    b = np.sqrt(6.0)
    x = torch.clamp(x, -b, b)
    t_cdf1 = (b-torch.abs(x))**2/12.
    t_cdf2 = 1 - t_cdf1
    flag0 = (x<-b).float()
    flag1 = ((x>=-b)*(x<0)).float()
    flag2 = ((x>=0)*(x<=b)).float()
    flag3 = (x>b).float()
    t_cdf = flag0 * zeros + flag1 * t_cdf1 + flag2 * t_cdf2 + flag3 * ones
    t_pdf = 1/b * (1-torch.abs(x)/b)
    t_pdf = flag0 * zeros + flag1 * t_pdf + flag2 * t_pdf + flag3 * zeros

    return t_pdf+1e-6, t_cdf

    #normal distribution
    normal = Normal(zeros, ones)
    g_cdf  = normal.cdf(x)
    g_pdf  = normal.log_prob(x).exp()+1e-6
    return g_pdf,g_cdf

    #mean Normal and Triangular distributions
    pdf = (t_pdf + g_pdf) * 0.5
    cdf = (t_cdf + g_cdf) * 0.5        
    return pdf+1e-6, cdf


def calc_mean_std(x):
    b, c = x.size()[:2]
    x = x.view(b, c, -1)
    feat_var = x.var(dim=-1) + 1e-6
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = x.mean(dim=-1).view(b, c, 1, 1)
    return feat_mean, feat_std
            
def commutativity_loss(x1, x2):
    x1_mean, x1_std = calc_mean_std(x1)
    x2_mean, x2_std = calc_mean_std(x2)
    pdf1, _ = calc_pdf_cdf((x1 - x1_mean) / x1_std)
    pdf2, _ = calc_pdf_cdf((x1 - x2_mean) / x2_std)
    pdf1to2, cdf1to2 = calc_pdf_cdf((x1 - x2_mean) / x2_std)
    pdf2to1, cdf2to1 = calc_pdf_cdf((x2 - x1_mean) / x1_std)
    #sgn1to2, sgn2to1 = (-1)**(cdf1to2>0.5).float(), (-1)**(cdf2to1>0.5).float()
    delta_pdf1to2 =  pdf2 - pdf1to2
    delta_pdf2to1 =  pdf1 - pdf2to1
    return F.mse_loss(delta_pdf1to2,delta_pdf2to1)


def sym(xs, x1, x2, label1, label2):
    '''
    xs:bchw
    x1:bchw
    x2:bchw
    label1:bchw
    label2:bchw
    '''
    b,c,h,w=xs.size()
    xs_mean, xs_std = calc_mean_std(xs)
    
    pdfs, _ = calc_pdf_cdf((xs - xs_mean) / xs_std)
    pdf1tos, cdf1tos = calc_pdf_cdf((x1 - xs_mean) / xs_std)
    pdf2tos, cdf2tos = calc_pdf_cdf((x2 - xs_mean) / xs_std)
    sgn1tos, sgn2tos = (-1)**(cdf1tos>0.5).float(), (-1)**(cdf2tos>0.5).float()
    delta_pdf1tos =  pdfs - pdf1tos
    delta_pdf2tos =  pdfs - pdf2tos


    #Feature-to-Label Difference Mapping
    delta_1, delta_2 = fc(gap(delta_pdf1tos * sgn1tos)), fc(gap(delta_pdf2tos * sgn2tos))  #线性卷积或者gap+fc,论文中的f函数

    pred1 = label2 - 2 * delta_1 
    pred2 = - label1 + 2 * delta_2
    

    loss={}
    loss['Ls'] = F.mse_loss(delta_pdf1tos,delta_pdf2tos) 
    loss['Lp'] = F.smooth_l1_loss(label1, pred1) + F.smooth_l1_loss(label2, pred2)
    loss['Lm'] = commutativity_loss(x1, x2)
    return loss



# Prior Set Selection
def get_template(num_template=128, shape=(512,512)):
    temp_dir='/data/config/awb/awb_bg/'
    fns = glob.glob(temp_dir+'online_search_pic/*/*.jpg')+glob.glob(temp_dir+'val2017/*.jpg')
   
    val_transform = transforms.Compose([
        transforms.Resize(shape),#,
        transforms.ToTensor(),
        ])
    numbers = random.sample(range(len(fns)), num_template)
    images = []
    for idx in numbers:
        rgb = Image.open(fns[idx]).convert('RGB')
        rgb = val_transform(rgb)
        images.append(rgb)
    images = torch.stack(images,0)
    return images
    
    

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
from taskcv_loader import CVDataLoader
from basenet1 import *
import torch.nn.functional as F
import os
from torch.nn.parallel import DataParallel
import scipy.io as sio 

#from aligned_reid.utils.utils import set_devices

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=24, train_path='/media/zrway/8T/HJK/aa/DomainNet/M3SDA/painting',
                 val_path='/media/zrway/8T/HJK/aa/DomainNet/M3SDA/sketch', learning_rate=0.0003, interval=100, optimizer='momentum'
                 , num_k=4, checkpoint_dir=None):
        self.train_path = args.train_path
        self.val_path = args.val_path
        self.num_k = args.num_k
        self.num_layer = args.num_layer
        self.batch_size = args.batch_size
        # print('batch_size:',batch_size)
        save_path = args.save+'_'+str(args.num_k)
        self.class_num= 345
        self.num_k1 = 8
        self.num_k2 = 1
        self.num_k3 = 8
        self.num_k4 = 1
        self.offset =0.1
        self.output_cr_t_C_label = np.zeros(batch_size)

        data_transforms = {
            train_path: transforms.Compose([
                transforms.Scale(256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            val_path: transforms.Compose([
                transforms.Scale(256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path,val_path]}
        #dset_s_sizes = {x: len(dsets[x]) for x in [train_path]}
        dset_s_sizes_dic = {x: len(dsets[x]) for x in [train_path]}
        dset_t_sizes_dic = {x: len(dsets[x]) for x in [val_path]}
        dset_s_sizes = list(dset_s_sizes_dic.values())[0]
        dset_t_sizes = list(dset_t_sizes_dic.values())[0]
        print("source_num")
        print(dset_s_sizes)
        print("target_num")
        print(dset_t_sizes)
        dset_classes = dsets[train_path].classes
        print ('classes'+str(dset_classes))
        use_gpu = torch.cuda.is_available()
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        train_loader = CVDataLoader()
        train_loader.initialize(dsets[train_path],dsets[val_path],batch_size)
        self.dataset = train_loader.load_data()
        test_loader = CVDataLoader()
        opt= args
        test_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=True)
        self.dataset_test = test_loader.load_data()
        option = 'resnet'+args.resnet

        self.G1 = ResNet_all(option)
        self.G2 = ResNet_all(option)
        self.G3 = ResNet_all(option)
        self.G4 = ResNet_all(option)

        #G_w = DataParallel(G)

        self.C = Predictor()#(num_layer=num_layer)
        self.C1 = Predictor()#(num_layer=num_layer)
        self.C2 = Predictor()#(num_layer=num_layer)
        self.D = AdversarialNetwork(2048)
        self.M = Mixer()
        gpus = args.gpu_id.split(',')
        if len(gpus) > 1:
            self.G1 = nn.DataParallel(G1, device_ids=[int(i) for i in gpus])
            self.G2 = nn.DataParallel(G2, device_ids=[int(i) for i in gpus])
            self.G3 = nn.DataParallel(G3, device_ids=[int(i) for i in gpus])
            self.G4 = nn.DataParallel(G4, device_ids=[int(i) for i in gpus])
            self.C = nn.DataParallel(C, device_ids=[int(i) for i in gpus])
            self.C1 = nn.DataParallel(C1, device_ids=[int(i) for i in gpus])
            self.C2 = nn.DataParallel(C2, device_ids=[int(i) for i in gpus])
            self.D = nn.DataParallel(D, device_ids=[int(i) for i in gpus])
            self.M = nn.DataParallel(M, device_ids=[int(i) for i in gpus])
        #D = Domain_discriminator()
        self.C.apply(weights_init)
        self.C1.apply(weights_init)
        self.C2.apply(weights_init)

        self.lr = args.lr


        if args.cuda:
            self.G1.cuda()
            self.G2.cuda()
            self.G3.cuda()
            self.G4.cuda()
            self.C.cuda()
            self.C1.cuda()
            self.C2.cuda()
            self.D.cuda()
            self.M.cuda()
        if args.optimizer == 'momentum':
            self.opt_g1 = optim.SGD(list(self.G1.features.parameters()), lr=args.lr,weight_decay=0.0005)
            self.opt_g2 = optim.SGD(list(self.G2.features.parameters()), lr=args.lr,weight_decay=0.0005)
            self.opt_g3 = optim.SGD(list(self.G3.features.parameters()), lr=args.lr,weight_decay=0.0005)
            self.opt_g4 = optim.SGD(list(self.G4.features.parameters()), lr=args.lr,weight_decay=0.0005)
            self.opt_c = optim.SGD(list(self.C.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
            self.opt_c1c2 = optim.SGD(list(self.C1.parameters())+list(self.C2.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
            self.opt_d = optim.SGD(list(self.D.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
            self.opt_m = optim.SGD(list(self.M.parameters()), lr=args.lr,weight_decay=0.0005)
        elif args.optimizer == 'adam':
            self.opt_g1 = optim.Adam(self.G1.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_g2 = optim.Adam(self.G2.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_g3 = optim.Adam(self.G3.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_g4 = optim.Adam(self.G4.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_c = optim.Adam(list(self.C.parameters()),lr=args.lr,weight_decay=0.0005)
            self.opt_c1c2 = optim.Adam(list(self.C1.parameters())+list(C2.parameters()), lr=args.lr,weight_decay=0.0005)
            self.opt_d = optim.Adam(list(self.D.parameters()),lr=args.lr,weight_decay=0.0005)
            self.opt_m = optim.Adam(list(self.M.parameters()), lr=args.lr,weight_decay=0.0005)
        else:
            self.opt_g1_cr = optim.Adadelta(self.G1.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_g2_cr = optim.Adadelta(self.G2.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_g3_cr = optim.Adadelta(self.G3.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_g4_cr = optim.Adadelta(self.G4.features.parameters(), lr=args.lr,weight_decay=0.0005)
            self.opt_c = optim.Adadelta(list(self.C.parameters()),lr=args.lr,weight_decay=0.0005)
            self.opt_c1c2 = optim.Adadelta(list(self.C1.parameters())+list(self.C2.parameters()),lr=args.lr,weight_decay=0.0005) 
            self.opt_d = optim.Adadelta(list(self.D.parameters()),lr=args.lr,weight_decay=0.0005)
            self.opt_m = optim.Adadelta(list(self.M.parameters()), lr=args.lr,weight_decay=0.0005)

    def reset_grad(self):
            self.opt_g1.zero_grad()
            self.opt_g2.zero_grad()
            self.opt_g3.zero_grad()
            self.opt_g4.zero_grad()
            self.opt_c.zero_grad()
            self.opt_c1c2.zero_grad()
            self.opt_d.zero_grad()
            self.opt_m.zero_grad()

    def discrepancy(self, out1, out2):
            return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


    def props_to_onehot(self, props):
           props = props.cpu().detach().numpy()
           if isinstance(props, list):
              props = np.array(props)
           a = np.argmax(props, axis=1)
           b = np.zeros((len(a), props.shape[1]))
           b[np.arange(len(a)), a] = 1
           return torch.from_numpy(b)

    def ent(self,output):
        out = -torch.mean(F.softmax(output.cuda() + 1e-6)*torch.log(F.softmax(output.cuda() + 1e-6)))
        out = Variable(out,requires_grad=True)
        return out

    def linear_mmd(self,f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss


    def train(self,num_epoch):
        criterion = nn.CrossEntropyLoss().cuda()
        adv_loss = nn.BCEWithLogitsLoss().cuda()
    
    ###############################################
        for ep in range(1,num_epoch):
            if ep ==2:
                min_J_w = max_J_w
            self.G1.train()
            self.G2.train()
            self.G3.train()
            self.G4.train()
            self.C.train()
            self.C1.train()
            self.C2.train()
            self.D.train()
            self.M.train()
            fea_for_LDA = np.empty(shape=(0,2048))
            fea_s_for_LDA = np.empty(shape=(0,2048)) 
            label_for_LDA = np.empty(shape=(0,1)) 
            label_s_for_LDA = []
            label_t_for_tSNE = []
  
        
            loss_mmd_all=0


            for batch_idx, data in enumerate(self.dataset):
                #if batch_idx * batch_size > 30000:
                 #break
                #if args.cuda:
                data1 = data['S']
                target1 = data['S_label']
                
                    #print('target1:',target1)
                data2  = data['T']
                target2 = data['T_label']
                data1, target1 = data1.cuda(), target1.cuda()
                data2 = data2.cuda()#, target2.cuda()
                # when pretraining network source only
                eta = 1.0
                data = Variable(torch.cat((data1,data2),0))
                target1 = Variable(target1)
            


        
################### source domain is discriminative.###################
                for i1 in range(self.num_k1):  
                    feat_cr_14=self.G1(data1)+self.G4(data1)
                    output_cr = self.C(feat_cr_14.cuda())
                
                    #print('output_cr:',output_cr)
                    loss_cr_ss = criterion(output_cr, target1)
                    loss_cr_ss.backward()
                    self.opt_g1.step()
                    self.opt_g4.step()
                    self.opt_c.step()
                    self.reset_grad()
#################### transferability ########################
                for i2 in range(self.num_k2):
                    feat_cr_s = self.G1(data1)
                    feat_cr_t = self.G1(data2)

                    output_cr_s_D = self.D(feat_cr_s.cuda())
                    output_cr_t_D = self.D(feat_cr_t.cuda())

                    loss_bce2 = nn.BCEWithLogitsLoss()(output_cr_s_D, output_cr_t_D.detach())
                    #loss_2 = 0.1*loss_bce2
                    loss_2 = 0.2*loss_bce2
                    loss_2.backward()
                    self.opt_d.step()
                    self.reset_grad()
            
#################### transferability (ds)########################
                    feat_cr_s1 = self.G1(data1)
                    feat_cr_t1 = self.G1(data2)
                    feat_cr_s2 = self.G2(data1)
                    feat_cr_t2 = self.G2(data2)
                    loss_33=self.discrepancy(feat_cr_s1,feat_cr_s2)+self.discrepancy(feat_cr_t1,feat_cr_t2)+self.discrepancy(feat_cr_s2,feat_cr_t2)
                    loss_3=-loss_33
                    loss_3.backward()
                    self.opt_g2.step()
                    self.reset_grad()

###################### discriminablity(ci) ##################
                    feat_cr_t3 = self.G3(data2)
                    #print('feat_cr_t3:',feat_cr_t3)
                    y=torch.exp(self.C(feat_cr_t3.cuda()))
                    #print('y是:',y)
                    loss_4=(1/self.class_num)*torch.log(y)
                    #loss_4=torch.log(self.C(feat_cr_t3.cuda()))
                    #print('loss_4:',loss_4)
                    loss_4.backward(loss_4.clone().detach())
                    self.opt_g3.step()
                    self.reset_grad()

###################### discriminablity(cs) ################## 
                    feat_cr_s4 = self.G4(data1)
                    feat_cr_t4 = self.G4(data2)
            
                    output_cr_s_C = self.C(feat_cr_s4.cuda())
                    output_cr_t_C = self.C(feat_cr_t4.cuda())
                    output_cr_s_C1 = self.C1(feat_cr_s4.cuda())
                    output_cr_s_C2 = self.C2(feat_cr_s4.cuda()) 
                    output_cr_t_C1 = self.C1(feat_cr_t4.cuda())
                    output_cr_t_C2 = self.C2(feat_cr_t4.cuda())


                    loss_cr_s = criterion(output_cr_s_C1, target1) + criterion(output_cr_s_C2, target1) + criterion(output_cr_s_C, target1)
                    loss_dis1_t = -self.discrepancy(output_cr_t_C1, output_cr_t_C2)
                    loss_5 = loss_cr_s + loss_dis1_t 
                    loss_5.backward()
                    self.opt_c1c2.step()
                    self.opt_c.step()
                    self.reset_grad()
                for i3 in range(self.num_k3):      
                    f1 = self.G1(data2)
                    f2 = self.G2(data2)
                    f3 = self.G3(data2)
                    f4 = self.G4(data2)
                    m_13=-self.discrepancy(self.M(f1), self.M(f3))
                    m_14=(self.discrepancy(self.M(f1), self.M(f4)))
                    m_23=self.discrepancy(self.M(f2), self.M(f3))
                    m_24=-self.discrepancy(self.M(f2), self.M(f4))
                    
                    loss_mal=-(m_13+m_14+m_23+m_24)
                    loss_mal.backward() 
                    self.opt_m.step()
                    self.reset_grad()

                for i4 in range(self.num_k4):
                    feat_cr_s4 = self.G4(data1)
                    feat_cr_t4 = self.G4(data2)
                    output_cr_s_D = self.D(feat_cr_s4.cuda())
                    output_cr_t_D = self.D(feat_cr_t4.cuda())
                    loss_bce1 = -nn.BCEWithLogitsLoss()(output_cr_s_D , output_cr_t_D.detach())
                    loss_6 = 0.2*loss_bce1
                    feat_cr_t4 = self.G4(data2)
                    output_cr_t_C = self.C(feat_cr_t4.cuda())
                    output_cr_t_C1 = self.C1(feat_cr_t4.cuda())
                    output_cr_t_C2 = self.C2(feat_cr_t4.cuda())
                    loss_71 = self.discrepancy(output_cr_t_C1, output_cr_t_C2)
                    loss_72 = self.discrepancy(output_cr_t_C, output_cr_t_C1)
                    loss_73 = self.discrepancy(output_cr_t_C, output_cr_t_C2)
                    loss_7 = loss_71 + loss_72 + loss_73
                    '''
                    feat_cr_s1 = self.G1(img_s)
                    feat_cr_t1 = self.G1(img_t)
                    feat_cr_s2 = self.G2(img_s)
                    feat_cr_t2 = self.G2(img_t)
                    loss_33=self.discrepancy(feat_cr_s1,feat_cr_s2)+self.discrepancy(feat_cr_t1,feat_cr_t2)+self.discrepancy(feat_cr_s2,feat_cr_t2)
                    loss_333=-loss_33
                
                    feat_cr_t3 = self.G3(img_t)
                    loss_44=(1/self.class_num)*torch.log(self.C(feat_cr_t3.cuda()))
                    '''
                    f1 = self.G1(data2)
                    f2 = self.G2(data2)
                    f3 = self.G3(data2)
                    f4 = self.G4(data2)
                    m_13=-self.discrepancy(self.M(f1), self.M(f3))
                    m_14=(self.discrepancy(self.M(f1), self.M(f4)))
                    m_23=self.discrepancy(self.M(f2), self.M(f3))
                    m_24=-self.discrepancy(self.M(f2), self.M(f4))
                    loss_mal=m_13+m_14+m_23+m_24
                
                    loss_all=loss_6+loss_7+loss_mal
                    #loss_all=loss_6+loss_333+loss_44+loss_7+loss_mal
                    #loss_all=loss_all.sum()
                    #print('loss_all:',loss_all)
                    #print('loss_6:',loss_6)
                    #print('loss_7:',loss_7)
                    #print('loss_mal:',loss_mal)
                    loss_all.backward()
                    self.opt_g1.step()
                    self.opt_g2.step()
                    self.opt_g3.step()
                    self.opt_g4.step()
                    self.reset_grad()
                
                    #print('epoch:,batch_idx:,loss_all:',ep, batch_idx,loss_all)
################ the balance of transferability and discriminability ###########
            return batch_idx
    	   
        ################################################
        #return A_st_min, A_st_max, min_Jw_s_1,max_J_w_1, A_st_norm, J_w_1_norm, batch_idx

        #-----------------end----------------------#         





    def test(self,epoch,acc):
        self.G1.eval()
        self.G2.eval()
        self.G3.eval()
        self.G4.eval()
        self.C.eval()
        self.C1.eval()
        self.C2.eval()
        self.D.eval()
        self.M.eval()
        test_loss1 = 0
        test_loss2 = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        correct5 = 0
        size = 0

        for batch_idx, data in enumerate(self.dataset_test):
            #if batch_idx*batch_size > 5000:
                #break
            #if args.cuda:
            data2  = data['T']
            target2 = data['T_label']
            #if val:
                #data2  = data['S']
                #target2 = data['S_label']
            data2, target2 = data2.cuda(), target2.cuda()
            data12, target12 = Variable(data2, volatile=True), Variable(target2)
            feat = self.G4(data12)
            output1 = self.C(feat)
            output21 = self.C1(feat)
            output22 = self.C2(feat)
            test_loss1 += F.nll_loss(output1, target12).item()
            test_loss2 += F.nll_loss(output21, target12).item()
            output_ensemble_c1c2 = output21 + output22
            output_ensemble_cc1c2 = output1 + output21 + output22
            pred1 = output1.data.max(1)[1]
            pred21 = output21.data.max(1)[1]
            pred22 = output22.data.max(1)[1]
            pred_ensemble_c1c2 = output_ensemble_c1c2.data.max(1)[1]
            pred_ensemble_cc1c2 = output_ensemble_cc1c2.data.max(1)[1]
            k = target12.data.size()[0]
            correct1 += pred1.eq(target12.data).cpu().sum()
            correct2 += pred21.eq(target12.data).cpu().sum()
            correct3 += pred22.eq(target12.data).cpu().sum()
            correct4 += pred_ensemble_c1c2.eq(target12.data).cpu().sum()
            correct5 += pred_ensemble_cc1c2.eq(target12.data).cpu().sum()
            size += k
        test_loss1 = test_loss1 / size
        test_loss2 = test_loss2 / size
        acc1 = 100. * float(correct1) / float(size)
        acc2 = 100. * float(correct2) / float(size)
        acc3 = 100. * float(correct3) / float(size)
        acc4 = 100. * float(correct4) / float(size)
        acc5 = 100. * float(correct5) / float(size)

        
        if max(acc1,acc2,acc3,acc4,acc5)>acc:
           acc = max(acc1,acc2,acc3,acc4,acc5)
           #acc00,acc11= acc1, value
           print( '\n Epoch: {}, Accuracy C: ({:.1f}%), Max Accuracy: ({:.1f}%)\n'.format(epoch, acc1, acc))
        #else:
           #print( '\n Epoch: {}, Accuracy C: ({:.1f}%), Max Accuracy: ({:.1f}%)\n'.format(epoch, acc1,acc))
       
    #if not val and value > 60:
    # if value > 98:
    #     torch.save(G.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'G.pth')
    #     torch.save(C.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'C.pth')
    #     torch.save(C1.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'C1.pth')
    #     torch.save(C2.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'C2.pth')
    #     torch.save(D.state_dict(), save_path+'_'+cross_domain+'_'+str(epoch)+'_'+str(value10)+'_'+'D.pth')
    #
        return acc
#for epoch in range(1, args.epochs + 1):
#train(args.epochs+1)

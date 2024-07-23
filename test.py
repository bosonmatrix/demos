import numpy as np
from tqdm import tqdm
import os
import pyproj
from sklearn import neighbors
import pandas as pd
import warnings

def load_order_file(order_file):
    with open(order_file, 'r') as file:
        lines = file.readlines()

    lines=list(filter(lambda x: x is not None, lines))

    order_info = {}
    image_info = {}

    for line in lines[1:10]:
        line = line.strip()
        if line:
            key_value = line.split(':',1)
            key = key_value[0].strip()
            if len(key_value) > 1:
                value = key_value[1].strip()
            else:
                value = None

            if key in order_info:
                order_info[key].append(value)
            else:
                order_info[key] = [value]

    for line in lines[11:-1]:
        line = line.strip()
        if line:
            key_value = line.split(':',1)
            key = key_value[0].strip()
            if len(key_value) > 1:
                value = key_value[1].strip()
                if key != 'ImageFile':
                    value=eval(value)
                else:
                    pass
            else:
                value = None

            if key in image_info:
                image_info[key].append(value)
            else:
                image_info[key] = [value]

    for key in image_info.keys():
        image_info[key]=np.asarray(image_info[key])

    return order_info,image_info

def load_image_tiept_data(tiept_file):
    with open(tiept_file, 'r') as file:
        lines = file.readlines()

    tiept_num=eval(lines[0])
    keys=['index','object_name','img_id','imgpt_y','imgpt_x']
    tiept_info = {}

    for line in tqdm(lines[1:-1],desc='loading tie points'):
        line = line.strip()

        if line:
            values = line.split('\t')
            for key_value in zip(keys,values):
                if key_value[0] not in tiept_info.keys():
                    tiept_info[key_value[0]]=[eval(key_value[1])]
                else:
                    tiept_info[key_value[0]].append(eval(key_value[1]))
                            
    for key in tiept_info.keys():
        tiept_info[key]=np.asarray(tiept_info[key])

    return tiept_num,tiept_info

def load_rpc_file(image_info):
    image_num=len(image_info['ImageID'])
    for i in tqdm(range(image_num),desc=f'loading RPC files'):
        (image_path,image_name)=os.path.split(image_info['ImageFile'][i])
        image_name_split=str.split(image_name,'.')
        rpc_file=os.path.join(image_path,image_name_split[0]+'.rpb')
        with open(rpc_file, 'r') as file:
            lines = file.readlines()

            for j in range(len(lines)):
                line = lines[j].strip()
                line_split=str.split(line,'=')
                if line_split[0].strip() not in ['satId','bandId','SpecId','BEGIN_GROUP']:
                    if line_split[0].strip()=='END_GROUP':
                        break
                    else:
                        key=line_split[0].strip()
                        if key not in ['lineNumCoef','lineDenCoef','sampNumCoef','sampDenCoef']:
                            if len(line_split)==1:
                                continue
                            else:
                                value=line_split[1].strip(';')
                                value=eval(value)
                                if key in image_info.keys():
                                    image_info[key].append(value)
                                else:
                                    image_info[key]=[value]
                        else:
                            begin_idx=j+1
                            end_idx=j+21
                            values=lines[begin_idx:end_idx]
                            values_pure=[each.strip() for each in values]
                            values_digital=[eval(each.strip((',;)'))) for each in values_pure]
                            if key in image_info.keys():
                                image_info[key].append(values_digital)
                            else:
                                image_info[key]=[values_digital]
                            
    return image_info

def forward_intersec_on_const_level(img_info,tiept_info):
    max_iter=100
    eps=1e-5

    current_tiept=dict.fromkeys(['index','object_name','img_id','imgpt_y','imgpt_x'])
    current_indices=np.where(np.isin(tiept_info['img_id'],img_info['ImageID']))
    for key in tiept_info.keys():
        current_tiept[key]=tiept_info[key][current_indices]

    objname_set,counts=np.unique(current_tiept['object_name'],return_counts=True)
    duplicate_indices=np.where(counts>1)[0]
    valid_objname=objname_set[duplicate_indices]
      
    tiept_idx = np.where(np.isin(current_tiept['object_name'],valid_objname))[0]

    for key in current_tiept.keys():
        temp_values=current_tiept[key][tiept_idx]
        current_tiept[key]=temp_values

    current_tiept['objpt_x']=np.zeros(len(tiept_idx))
    current_tiept['objpt_y']=np.zeros(len(tiept_idx))
    current_tiept['objpt_z']=np.zeros(len(tiept_idx))

    for obj_name in tqdm(valid_objname):
        indices=np.where(current_tiept['object_name']==obj_name)[0]
        tiept_num=len(indices)
        for idx in indices:
            flag=True
            
            image_id=current_tiept['img_id'][idx]

            X=np.zeros((2))
            A=np.zeros((2,2))
            f=np.zeros((2))
            P=np.eye(2)

            iter_count=0

            lon=img_info['longOffset'][image_id]
            lat=img_info['latOffset'][image_id]
            hei=img_info['heightOffset'][image_id]

            while flag:
                tqdm.write(f'the {iter_count+1}th iteration of tiepoint {obj_name}')
                
                L=(lon-img_info['longOffset'][image_id])/img_info['longScale'][image_id]
                B=(lat-img_info['latOffset'][image_id])/img_info['latScale'][image_id]
                H=(hei-img_info['heightOffset'][image_id])/img_info['heightScale'][image_id]

                BL=B*L;BH=B*H;LH=L*H;BB=B*B;LL=L*L;HH=H*H;BLH=B*L*H;BBL=B*B*L;BBH=B*B*H;LLB=L*L*B;LLH=L*L*H;HHB=H*H*B;HHL=H*H*L;BBB=B*B*B;LLL=L*L*L;HHH=H*H*H
                LNUM=img_info['lineNumCoef'][image_id][0]+img_info['lineNumCoef'][image_id][1]*L+img_info['lineNumCoef'][image_id][2]*B+img_info['lineNumCoef'][image_id][3]*H\
                    +img_info['lineNumCoef'][image_id][4]*BL+img_info['lineNumCoef'][image_id][5]*LH+img_info['lineNumCoef'][image_id][6]*BH\
                    +img_info['lineNumCoef'][image_id][7]*LL+img_info['lineNumCoef'][image_id][8]*BB+img_info['lineNumCoef'][image_id][9]*HH\
                    +img_info['lineNumCoef'][image_id][10]*BLH+img_info['lineNumCoef'][image_id][11]*LLL+img_info['lineNumCoef'][image_id][12]*BBL\
                    +img_info['lineNumCoef'][image_id][13]*HHL+img_info['lineNumCoef'][image_id][14]*LLB+img_info['lineNumCoef'][image_id][15]*BBB\
                    +img_info['lineNumCoef'][image_id][16]*HHB+img_info['lineNumCoef'][image_id][17]*LLH+img_info['lineNumCoef'][image_id][18]*BBH\
                    +img_info['lineNumCoef'][image_id][19]*HHH
            
                LDEN=img_info['lineDenCoef'][image_id][0]+img_info['lineDenCoef'][image_id][1]*L+img_info['lineDenCoef'][image_id][2]*B+img_info['lineDenCoef'][image_id][3]*H\
                        +img_info['lineDenCoef'][image_id][4]*BL+img_info['lineDenCoef'][image_id][5]*LH+img_info['lineDenCoef'][image_id][6]*BH\
                        +img_info['lineDenCoef'][image_id][7]*LL+img_info['lineDenCoef'][image_id][8]*BB+img_info['lineDenCoef'][image_id][9]*HH\
                        +img_info['lineDenCoef'][image_id][10]*BLH+img_info['lineDenCoef'][image_id][11]*LLL+img_info['lineDenCoef'][image_id][12]*BBL\
                        +img_info['lineDenCoef'][image_id][13]*HHL+img_info['lineDenCoef'][image_id][14]*LLB+img_info['lineDenCoef'][image_id][15]*BBB\
                        +img_info['lineDenCoef'][image_id][16]*HHB+img_info['lineDenCoef'][image_id][17]*LLH+img_info['lineDenCoef'][image_id][18]*BBH\
                        +img_info['lineDenCoef'][image_id][19]*HHH
                        
                SNUM=img_info['sampNumCoef'][image_id][0]+img_info['sampNumCoef'][image_id][1]*L+img_info['sampNumCoef'][image_id][2]*B+img_info['sampNumCoef'][image_id][3]*H\
                        +img_info['sampNumCoef'][image_id][4]*BL+img_info['sampNumCoef'][image_id][5]*LH+img_info['sampNumCoef'][image_id][6]*BH\
                        +img_info['sampNumCoef'][image_id][7]*LL+img_info['sampNumCoef'][image_id][8]*BB+img_info['sampNumCoef'][image_id][9]*HH\
                        +img_info['sampNumCoef'][image_id][10]*BLH+img_info['sampNumCoef'][image_id][11]*LLL+img_info['sampNumCoef'][image_id][12]*BBL\
                        +img_info['sampNumCoef'][image_id][13]*HHL+img_info['sampNumCoef'][image_id][14]*LLB+img_info['sampNumCoef'][image_id][15]*BBB\
                        +img_info['sampNumCoef'][image_id][16]*HHB+img_info['sampNumCoef'][image_id][17]*LLH+img_info['sampNumCoef'][image_id][18]*BBH\
                        +img_info['sampNumCoef'][image_id][19]*HHH
                
                SDEN=img_info['sampDenCoef'][image_id][0]+img_info['sampDenCoef'][image_id][1]*L+img_info['sampDenCoef'][image_id][2]*B+img_info['sampDenCoef'][image_id][3]*H\
                        +img_info['sampDenCoef'][image_id][4]*BL+img_info['sampDenCoef'][image_id][5]*LH+img_info['sampDenCoef'][image_id][6]*BH\
                        +img_info['sampDenCoef'][image_id][7]*LL+img_info['sampDenCoef'][image_id][8]*BB+img_info['sampDenCoef'][image_id][9]*HH\
                        +img_info['sampDenCoef'][image_id][10]*BLH+img_info['sampDenCoef'][image_id][11]*LLL+img_info['sampDenCoef'][image_id][12]*BBL\
                        +img_info['sampDenCoef'][image_id][13]*HHL+img_info['sampDenCoef'][image_id][14]*LLB+img_info['sampDenCoef'][image_id][15]*BBB\
                        +img_info['sampDenCoef'][image_id][16]*HHB+img_info['sampDenCoef'][image_id][17]*LLH+img_info['sampDenCoef'][image_id][18]*BBH\
                        +img_info['sampDenCoef'][image_id][19]*HHH
                
                dLNUMB=img_info['lineNumCoef'][image_id][2]+img_info['lineNumCoef'][image_id][4]*L+img_info['lineNumCoef'][image_id][6]*H+2*img_info['lineNumCoef'][image_id][8]*B+img_info['lineNumCoef'][image_id][10]*LH+2*img_info['lineNumCoef'][image_id][12]*BL+\
                img_info['lineNumCoef'][image_id][14]*LL+3*img_info['lineNumCoef'][image_id][15]*BB+img_info['lineNumCoef'][image_id][16]*HH+2*img_info['lineNumCoef'][image_id][18]*BH
                dLNUML=img_info['lineNumCoef'][image_id][1]+img_info['lineNumCoef'][image_id][4]*B+img_info['lineNumCoef'][image_id][5]*H+2*img_info['lineNumCoef'][image_id][7]*L+img_info['lineNumCoef'][image_id][10]*BH\
                    +3*img_info['lineNumCoef'][image_id][11]*LL+img_info['lineNumCoef'][image_id][12]*BB+img_info['lineNumCoef'][image_id][13]*HH+2*img_info['lineNumCoef'][image_id][14]*BL+2*img_info['lineNumCoef'][image_id][17]*H*L
                dLNUMH=img_info['lineNumCoef'][image_id][3]+img_info['lineNumCoef'][image_id][5]*L+img_info['lineNumCoef'][image_id][6]*B+2*img_info['lineNumCoef'][image_id][9]*H\
                    +img_info['lineNumCoef'][image_id][10]*BL+2*img_info['lineNumCoef'][image_id][13]*L*H+2*img_info['lineNumCoef'][image_id][16]*B*H+img_info['lineNumCoef'][image_id][17]*LL+img_info['lineNumCoef'][image_id][18]*BB+3*img_info['lineNumCoef'][image_id][19]*HH

                dLDENB=img_info['lineDenCoef'][image_id][2]+img_info['lineDenCoef'][image_id][4]*L+img_info['lineDenCoef'][image_id][6]*H+2*img_info['lineDenCoef'][image_id][8]*B+img_info['lineDenCoef'][image_id][10]*LH+2*img_info['lineDenCoef'][image_id][12]*BL+\
                    img_info['lineDenCoef'][image_id][14]*LL+3*img_info['lineDenCoef'][image_id][15]*BB+img_info['lineDenCoef'][image_id][16]*HH+2*img_info['lineDenCoef'][image_id][18]*BH
                dLDENL=img_info['lineDenCoef'][image_id][1]+img_info['lineDenCoef'][image_id][4]*B+img_info['lineDenCoef'][image_id][5]*H+2*img_info['lineDenCoef'][image_id][7]*L+img_info['lineDenCoef'][image_id][10]*BH\
                    +3*img_info['lineDenCoef'][image_id][11]*LL+img_info['lineDenCoef'][image_id][12]*BB+img_info['lineDenCoef'][image_id][13]*HH+2*img_info['lineDenCoef'][image_id][14]*BL+2*img_info['lineDenCoef'][image_id][17]*H*L
                dLDENH=img_info['lineDenCoef'][image_id][3]+img_info['lineDenCoef'][image_id][5]*L+img_info['lineDenCoef'][image_id][6]*B+2*img_info['lineDenCoef'][image_id][9]*H\
                    +img_info['lineDenCoef'][image_id][10]*BL+2*img_info['lineDenCoef'][image_id][13]*L*H+2*img_info['lineDenCoef'][image_id][16]*B*H+img_info['lineDenCoef'][image_id][17]*LL+img_info['lineDenCoef'][image_id][18]*BB+3*img_info['lineDenCoef'][image_id][19]*HH

                dSNUMB=img_info['sampNumCoef'][image_id][2]+img_info['sampNumCoef'][image_id][4]*L+img_info['sampNumCoef'][image_id][6]*H+2*img_info['sampNumCoef'][image_id][8]*B+img_info['sampNumCoef'][image_id][10]*LH+2*img_info['sampNumCoef'][image_id][12]*BL+\
                    img_info['sampNumCoef'][image_id][14]*LL+3*img_info['sampNumCoef'][image_id][15]*BB+img_info['sampNumCoef'][image_id][16]*HH+2*img_info['sampNumCoef'][image_id][18]*BH
                dSNUML=img_info['sampNumCoef'][image_id][1]+img_info['sampNumCoef'][image_id][4]*B+img_info['sampNumCoef'][image_id][5]*H+2*img_info['sampNumCoef'][image_id][7]*L+img_info['sampNumCoef'][image_id][10]*BH\
                    +3*img_info['sampNumCoef'][image_id][11]*LL+img_info['sampNumCoef'][image_id][12]*BB+img_info['sampNumCoef'][image_id][13]*HH+2*img_info['sampNumCoef'][image_id][14]*BL+2*img_info['sampNumCoef'][image_id][17]*H*L
                dSNUMH=img_info['sampNumCoef'][image_id][3]+img_info['sampNumCoef'][image_id][5]*L+img_info['sampNumCoef'][image_id][6]*B+2*img_info['sampNumCoef'][image_id][9]*H\
                    +img_info['sampNumCoef'][image_id][10]*BL+2*img_info['sampNumCoef'][image_id][13]*L*H+2*img_info['sampNumCoef'][image_id][16]*B*H+img_info['sampNumCoef'][image_id][17]*LL+img_info['sampNumCoef'][image_id][18]*BB+3*img_info['sampNumCoef'][image_id][19]*HH

                dSDENB=img_info['sampDenCoef'][image_id][2]+img_info['sampDenCoef'][image_id][4]*L+img_info['sampDenCoef'][image_id][6]*H+2*img_info['sampDenCoef'][image_id][8]*B+img_info['sampDenCoef'][image_id][10]*LH+2*img_info['sampDenCoef'][image_id][12]*BL+\
                    img_info['sampDenCoef'][image_id][14]*LL+3*img_info['sampDenCoef'][image_id][15]*BB+img_info['sampDenCoef'][image_id][16]*HH+2*img_info['sampDenCoef'][image_id][18]*BH
                dSDENL=img_info['sampDenCoef'][image_id][1]+img_info['sampDenCoef'][image_id][4]*B+img_info['sampDenCoef'][image_id][5]*H+2*img_info['sampDenCoef'][image_id][7]*L+img_info['sampDenCoef'][image_id][10]*BH\
                    +3*img_info['sampDenCoef'][image_id][11]*LL+img_info['sampDenCoef'][image_id][12]*BB+img_info['sampDenCoef'][image_id][13]*HH+2*img_info['sampDenCoef'][image_id][14]*BL+2*img_info['sampDenCoef'][image_id][17]*H*L
                dSDENH=img_info['sampDenCoef'][image_id][3]+img_info['sampDenCoef'][image_id][5]*L+img_info['sampDenCoef'][image_id][6]*B+2*img_info['sampDenCoef'][image_id][9]*H\
                    +img_info['sampDenCoef'][image_id][10]*BL+2*img_info['sampDenCoef'][image_id][13]*L*H+2*img_info['sampDenCoef'][image_id][16]*B*H+img_info['sampDenCoef'][image_id][17]*LL+img_info['sampDenCoef'][image_id][18]*BB+3*img_info['sampDenCoef'][image_id][19]*HH

                dLB=(dLNUMB*LDEN-LNUM*dLDENB)/(LDEN*LDEN)
                dLL=(dLNUML*LDEN-LNUM*dLDENL)/(LDEN*LDEN)
                dLH=(dLNUMH*LDEN-LNUM*dLDENH)/(LDEN*LDEN)

                dSB=(dSNUMB*SDEN-SNUM*dSDENB)/(SDEN*SDEN)
                dSL=(dSNUML*SDEN-SNUM*dSDENL)/(SDEN*SDEN)
                dSH=(dSNUMH*SDEN-SNUM*dSDENH)/(SDEN*SDEN)

                dLlat=dLB/img_info['latScale'][image_id];dLlon=dLL/img_info['longScale'][image_id];dLhei=dLH/img_info['heightScale'][image_id]
                dSlat=dSB/img_info['latScale'][image_id];dSlon=dSL/img_info['longScale'][image_id];dShei=dSH/img_info['heightScale'][image_id]

                l=np.asarray(current_tiept['imgpt_x'][idx])
                s=np.asarray(current_tiept['imgpt_y'][idx])

                A=np.asarray([[dLlon*img_info['lineScale'][image_id],dLlat*img_info['lineScale'][image_id]],\
                                                                [dSlon*img_info['sampScale'][image_id],dSlat*img_info['sampScale'][image_id]]])
                fl_0=(LNUM/LDEN)*img_info['lineScale'][image_id]+img_info['lineOffset'][image_id]
                fs_0=(SNUM/SDEN)*img_info['sampScale'][image_id]+img_info['sampOffset'][image_id]
                                
                f[0]=l-fl_0
                f[1]=s-fs_0
                f=f

                coeff=A.T@P@A
                Q_xx=np.linalg.inv(coeff)
                adj_esti=Q_xx@A.T@P@f

                X=adj_esti

                iter_count+=1

                x_length=len(X)
                lon+=X[0]
                lat+=X[1]

                if np.fabs(np.max(adj_esti))<=eps or iter_count>=max_iter:
                    flag=False

            current_tiept['objpt_x'][idx]=lon
            current_tiept['objpt_y'][idx]=lat
            current_tiept['objpt_z'][idx]=hei
            
    return current_tiept

def forward_intersec(img_info,tiept_info):
    max_iter=100
    eps=1e-5

    current_tiept=tiept_info

    objname_set,counts=np.unique(current_tiept['object_name'],return_counts=True)
    duplicate_indices=np.where(counts>1)[0]
    valid_objname=objname_set[duplicate_indices]
    
    for obj_name in tqdm(valid_objname,desc='forward intersecting'):
        flag=True
        indices=np.where(current_tiept['object_name']==obj_name)[0]
        tiept_num=len(indices)
        image_id=current_tiept['img_id'][indices]

        X=np.zeros((3*tiept_num))
        A=np.zeros((2*tiept_num,3))
        f=np.zeros((2*tiept_num))
        P=np.eye(2*tiept_num)

        iter_count=0

        lon=current_tiept['objpt_x'][indices]
        lat=current_tiept['objpt_y'][indices]
        hei=current_tiept['objpt_z'][indices]

        while flag:
            tqdm.write(f'the {iter_count+1}th iteration of tiepoint {obj_name}')
            
            L=(lon-img_info['longOffset'][image_id])/img_info['longScale'][image_id]
            B=(lat-img_info['latOffset'][image_id])/img_info['latScale'][image_id]
            H=(hei-img_info['heightOffset'][image_id])/img_info['heightScale'][image_id]

            BL=B*L;BH=B*H;LH=L*H;BB=B*B;LL=L*L;HH=H*H;BLH=B*L*H;BBL=B*B*L;BBH=B*B*H;LLB=L*L*B;LLH=L*L*H;HHB=H*H*B;HHL=H*H*L;BBB=B*B*B;LLL=L*L*L;HHH=H*H*H
            LNUM=img_info['lineNumCoef'][image_id][:,0]+img_info['lineNumCoef'][image_id][:,1]*L+img_info['lineNumCoef'][image_id][:,2]*B+img_info['lineNumCoef'][image_id][:,3]*H\
                    +img_info['lineNumCoef'][image_id][:,4]*BL+img_info['lineNumCoef'][image_id][:,5]*LH+img_info['lineNumCoef'][image_id][:,6]*BH\
                    +img_info['lineNumCoef'][image_id][:,7]*LL+img_info['lineNumCoef'][image_id][:,8]*BB+img_info['lineNumCoef'][image_id][:,9]*HH\
                    +img_info['lineNumCoef'][image_id][:,10]*BLH+img_info['lineNumCoef'][image_id][:,11]*LLL+img_info['lineNumCoef'][image_id][:,12]*BBL\
                    +img_info['lineNumCoef'][image_id][:,13]*HHL+img_info['lineNumCoef'][image_id][:,14]*LLB+img_info['lineNumCoef'][image_id][:,15]*BBB\
                    +img_info['lineNumCoef'][image_id][:,16]*HHB+img_info['lineNumCoef'][image_id][:,17]*LLH+img_info['lineNumCoef'][image_id][:,18]*BBH\
                    +img_info['lineNumCoef'][image_id][:,19]*HHH
            
            LDEN=img_info['lineDenCoef'][image_id][:,0]+img_info['lineDenCoef'][image_id][:,1]*L+img_info['lineDenCoef'][image_id][:,2]*B+img_info['lineDenCoef'][image_id][:,3]*H\
                    +img_info['lineDenCoef'][image_id][:,4]*BL+img_info['lineDenCoef'][image_id][:,5]*LH+img_info['lineDenCoef'][image_id][:,6]*BH\
                    +img_info['lineDenCoef'][image_id][:,7]*LL+img_info['lineDenCoef'][image_id][:,8]*BB+img_info['lineDenCoef'][image_id][:,9]*HH\
                    +img_info['lineDenCoef'][image_id][:,10]*BLH+img_info['lineDenCoef'][image_id][:,11]*LLL+img_info['lineDenCoef'][image_id][:,12]*BBL\
                    +img_info['lineDenCoef'][image_id][:,13]*HHL+img_info['lineDenCoef'][image_id][:,14]*LLB+img_info['lineDenCoef'][image_id][:,15]*BBB\
                    +img_info['lineDenCoef'][image_id][:,16]*HHB+img_info['lineDenCoef'][image_id][:,17]*LLH+img_info['lineDenCoef'][image_id][:,18]*BBH\
                    +img_info['lineDenCoef'][image_id][:,19]*HHH
                    
            SNUM=img_info['sampNumCoef'][image_id][:,0]+img_info['sampNumCoef'][image_id][:,1]*L+img_info['sampNumCoef'][image_id][:,2]*B+img_info['sampNumCoef'][image_id][:,3]*H\
                    +img_info['sampNumCoef'][image_id][:,4]*BL+img_info['sampNumCoef'][image_id][:,5]*LH+img_info['sampNumCoef'][image_id][:,6]*BH\
                    +img_info['sampNumCoef'][image_id][:,7]*LL+img_info['sampNumCoef'][image_id][:,8]*BB+img_info['sampNumCoef'][image_id][:,9]*HH\
                    +img_info['sampNumCoef'][image_id][:,10]*BLH+img_info['sampNumCoef'][image_id][:,11]*LLL+img_info['sampNumCoef'][image_id][:,12]*BBL\
                    +img_info['sampNumCoef'][image_id][:,13]*HHL+img_info['sampNumCoef'][image_id][:,14]*LLB+img_info['sampNumCoef'][image_id][:,15]*BBB\
                    +img_info['sampNumCoef'][image_id][:,16]*HHB+img_info['sampNumCoef'][image_id][:,17]*LLH+img_info['sampNumCoef'][image_id][:,18]*BBH\
                    +img_info['sampNumCoef'][image_id][:,19]*HHH
            
            SDEN=img_info['sampDenCoef'][image_id][:,0]+img_info['sampDenCoef'][image_id][:,1]*L+img_info['sampDenCoef'][image_id][:,2]*B+img_info['sampDenCoef'][image_id][:,3]*H\
                    +img_info['sampDenCoef'][image_id][:,4]*BL+img_info['sampDenCoef'][image_id][:,5]*LH+img_info['sampDenCoef'][image_id][:,6]*BH\
                    +img_info['sampDenCoef'][image_id][:,7]*LL+img_info['sampDenCoef'][image_id][:,8]*BB+img_info['sampDenCoef'][image_id][:,9]*HH\
                    +img_info['sampDenCoef'][image_id][:,10]*BLH+img_info['sampDenCoef'][image_id][:,11]*LLL+img_info['sampDenCoef'][image_id][:,12]*BBL\
                    +img_info['sampDenCoef'][image_id][:,13]*HHL+img_info['sampDenCoef'][image_id][:,14]*LLB+img_info['sampDenCoef'][image_id][:,15]*BBB\
                    +img_info['sampDenCoef'][image_id][:,16]*HHB+img_info['sampDenCoef'][image_id][:,17]*LLH+img_info['sampDenCoef'][image_id][:,18]*BBH\
                    +img_info['sampDenCoef'][image_id][:,19]*HHH
            
            dLNUMB=img_info['lineNumCoef'][image_id][:,2]+img_info['lineNumCoef'][image_id][:,4]*L+img_info['lineNumCoef'][image_id][:,6]*H+2*img_info['lineNumCoef'][image_id][:,8]*B+img_info['lineNumCoef'][image_id][:,10]*LH+2*img_info['lineNumCoef'][image_id][:,12]*BL+\
                img_info['lineNumCoef'][image_id][:,14]*LL+3*img_info['lineNumCoef'][image_id][:,15]*BB+img_info['lineNumCoef'][image_id][:,16]*HH+2*img_info['lineNumCoef'][image_id][:,18]*BH
            dLNUML=img_info['lineNumCoef'][image_id][:,1]+img_info['lineNumCoef'][image_id][:,4]*B+img_info['lineNumCoef'][image_id][:,5]*H+2*img_info['lineNumCoef'][image_id][:,7]*L+img_info['lineNumCoef'][image_id][:,10]*BH\
                +3*img_info['lineNumCoef'][image_id][:,11]*LL+img_info['lineNumCoef'][image_id][:,12]*BB+img_info['lineNumCoef'][image_id][:,13]*HH+2*img_info['lineNumCoef'][image_id][:,14]*BL+2*img_info['lineNumCoef'][image_id][:,17]*H*L
            dLNUMH=img_info['lineNumCoef'][image_id][:,3]+img_info['lineNumCoef'][image_id][:,5]*L+img_info['lineNumCoef'][image_id][:,6]*B+2*img_info['lineNumCoef'][image_id][:,9]*H\
                +img_info['lineNumCoef'][image_id][:,10]*BL+2*img_info['lineNumCoef'][image_id][:,13]*L*H+2*img_info['lineNumCoef'][image_id][:,16]*B*H+img_info['lineNumCoef'][image_id][:,17]*LL+img_info['lineNumCoef'][image_id][:,18]*BB+3*img_info['lineNumCoef'][image_id][:,19]*HH

            dLDENB=img_info['lineDenCoef'][image_id][:,2]+img_info['lineDenCoef'][image_id][:,4]*L+img_info['lineDenCoef'][image_id][:,6]*H+2*img_info['lineDenCoef'][image_id][:,8]*B+img_info['lineDenCoef'][image_id][:,10]*LH+2*img_info['lineDenCoef'][image_id][:,12]*BL+\
                img_info['lineDenCoef'][image_id][:,14]*LL+3*img_info['lineDenCoef'][image_id][:,15]*BB+img_info['lineDenCoef'][image_id][:,16]*HH+2*img_info['lineDenCoef'][image_id][:,18]*BH
            dLDENL=img_info['lineDenCoef'][image_id][:,1]+img_info['lineDenCoef'][image_id][:,4]*B+img_info['lineDenCoef'][image_id][:,5]*H+2*img_info['lineDenCoef'][image_id][:,7]*L+img_info['lineDenCoef'][image_id][:,10]*BH\
                +3*img_info['lineDenCoef'][image_id][:,11]*LL+img_info['lineDenCoef'][image_id][:,12]*BB+img_info['lineDenCoef'][image_id][:,13]*HH+2*img_info['lineDenCoef'][image_id][:,14]*BL+2*img_info['lineDenCoef'][image_id][:,17]*H*L
            dLDENH=img_info['lineDenCoef'][image_id][:,3]+img_info['lineDenCoef'][image_id][:,5]*L+img_info['lineDenCoef'][image_id][:,6]*B+2*img_info['lineDenCoef'][image_id][:,9]*H\
                +img_info['lineDenCoef'][image_id][:,10]*BL+2*img_info['lineDenCoef'][image_id][:,13]*L*H+2*img_info['lineDenCoef'][image_id][:,16]*B*H+img_info['lineDenCoef'][image_id][:,17]*LL+img_info['lineDenCoef'][image_id][:,18]*BB+3*img_info['lineDenCoef'][image_id][:,19]*HH

            dSNUMB=img_info['sampNumCoef'][image_id][:,2]+img_info['sampNumCoef'][image_id][:,4]*L+img_info['sampNumCoef'][image_id][:,6]*H+2*img_info['sampNumCoef'][image_id][:,8]*B+img_info['sampNumCoef'][image_id][:,10]*LH+2*img_info['sampNumCoef'][image_id][:,12]*BL+\
                img_info['sampNumCoef'][image_id][:,14]*LL+3*img_info['sampNumCoef'][image_id][:,15]*BB+img_info['sampNumCoef'][image_id][:,16]*HH+2*img_info['sampNumCoef'][image_id][:,18]*BH
            dSNUML=img_info['sampNumCoef'][image_id][:,1]+img_info['sampNumCoef'][image_id][:,4]*B+img_info['sampNumCoef'][image_id][:,5]*H+2*img_info['sampNumCoef'][image_id][:,7]*L+img_info['sampNumCoef'][image_id][:,10]*BH\
                +3*img_info['sampNumCoef'][image_id][:,11]*LL+img_info['sampNumCoef'][image_id][:,12]*BB+img_info['sampNumCoef'][image_id][:,13]*HH+2*img_info['sampNumCoef'][image_id][:,14]*BL+2*img_info['sampNumCoef'][image_id][:,17]*H*L
            dSNUMH=img_info['sampNumCoef'][image_id][:,3]+img_info['sampNumCoef'][image_id][:,5]*L+img_info['sampNumCoef'][image_id][:,6]*B+2*img_info['sampNumCoef'][image_id][:,9]*H\
                +img_info['sampNumCoef'][image_id][:,10]*BL+2*img_info['sampNumCoef'][image_id][:,13]*L*H+2*img_info['sampNumCoef'][image_id][:,16]*B*H+img_info['sampNumCoef'][image_id][:,17]*LL+img_info['sampNumCoef'][image_id][:,18]*BB+3*img_info['sampNumCoef'][image_id][:,19]*HH

            dSDENB=img_info['sampDenCoef'][image_id][:,2]+img_info['sampDenCoef'][image_id][:,4]*L+img_info['sampDenCoef'][image_id][:,6]*H+2*img_info['sampDenCoef'][image_id][:,8]*B+img_info['sampDenCoef'][image_id][:,10]*LH+2*img_info['sampDenCoef'][image_id][:,12]*BL+\
                img_info['sampDenCoef'][image_id][:,14]*LL+3*img_info['sampDenCoef'][image_id][:,15]*BB+img_info['sampDenCoef'][image_id][:,16]*HH+2*img_info['sampDenCoef'][image_id][:,18]*BH
            dSDENL=img_info['sampDenCoef'][image_id][:,1]+img_info['sampDenCoef'][image_id][:,4]*B+img_info['sampDenCoef'][image_id][:,5]*H+2*img_info['sampDenCoef'][image_id][:,7]*L+img_info['sampDenCoef'][image_id][:,10]*BH\
                +3*img_info['sampDenCoef'][image_id][:,11]*LL+img_info['sampDenCoef'][image_id][:,12]*BB+img_info['sampDenCoef'][image_id][:,13]*HH+2*img_info['sampDenCoef'][image_id][:,14]*BL+2*img_info['sampDenCoef'][image_id][:,17]*H*L
            dSDENH=img_info['sampDenCoef'][image_id][:,3]+img_info['sampDenCoef'][image_id][:,5]*L+img_info['sampDenCoef'][image_id][:,6]*B+2*img_info['sampDenCoef'][image_id][:,9]*H\
                +img_info['sampDenCoef'][image_id][:,10]*BL+2*img_info['sampDenCoef'][image_id][:,13]*L*H+2*img_info['sampDenCoef'][image_id][:,16]*B*H+img_info['sampDenCoef'][image_id][:,17]*LL+img_info['sampDenCoef'][image_id][:,18]*BB+3*img_info['sampDenCoef'][image_id][:,19]*HH

            dLB=(dLNUMB*LDEN-LNUM*dLDENB)/(LDEN*LDEN)
            dLL=(dLNUML*LDEN-LNUM*dLDENL)/(LDEN*LDEN)
            dLH=(dLNUMH*LDEN-LNUM*dLDENH)/(LDEN*LDEN)

            dSB=(dSNUMB*SDEN-SNUM*dSDENB)/(SDEN*SDEN)
            dSL=(dSNUML*SDEN-SNUM*dSDENL)/(SDEN*SDEN)
            dSH=(dSNUMH*SDEN-SNUM*dSDENH)/(SDEN*SDEN)

            dLlat=dLB/img_info['latScale'][image_id];dLlon=dLL/img_info['longScale'][image_id];dLhei=dLH/img_info['heightScale'][image_id]
            dSlat=dSB/img_info['latScale'][image_id];dSlon=dSL/img_info['longScale'][image_id];dShei=dSH/img_info['heightScale'][image_id]

            l=np.asarray(current_tiept['imgpt_x'][indices])
            s=np.asarray(current_tiept['imgpt_y'][indices])
            # l=l+line_refine
            # s=s+sample_refine

            for i in range(tiept_num):
                A[i*2:(i+1)*2,:]=np.asarray([[dLlon[i]*img_info['lineScale'][image_id][i],dLlat[i]*img_info['lineScale'][image_id][i],dLhei[i]*img_info['lineScale'][image_id][i]],\
                                                                [dSlon[i]*img_info['sampScale'][image_id][i],dSlat[i]*img_info['sampScale'][image_id][i],dShei[i]*img_info['sampScale'][image_id][i]]])
                fl_0=(LNUM[i]/LDEN[i])*img_info['lineScale'][i]+img_info['lineOffset'][i]
                fs_0=(SNUM[i]/SDEN[i])*img_info['sampScale'][i]+img_info['sampOffset'][i]
                                
                f[i*2]=l[i]-fl_0
                f[i*2+1]=s[i]-fs_0

            coeff=A.T@P@A
            Q_xx=np.linalg.inv(coeff)
            adj_esti=Q_xx@A.T@P@f

            X=adj_esti

            iter_count+=1

            if np.fabs(np.max(adj_esti))<=eps or iter_count>=max_iter:
                flag=False
            else:
                lon+=X[0]
                lat+=X[1]
                hei+=X[2]

                current_tiept['objpt_x'][indices]=lon
                current_tiept['objpt_y'][indices]=lat
                current_tiept['objpt_z'][indices]=hei

    return current_tiept

if __name__=='__main__':
    order_file=r'F:\phD_career\multi_source_adjustment\data\guangzhou-demo\auxiliary\zdb.tri.txt'
    tiept_file=r'F:\phD_career\multi_source_adjustment\data\guangzhou-demo\auxiliary\zdb.tiepick-ties.tie'
    SLA_path=r'/data3/weiyu/ReferenceRSData/GlobalData/Laser/ICESat-2/ATL08_GroundHeightControlPoint/ICESat2_2020_V1'
    rpc_path=r'F:\phD_career\multi_source_adjustment\data\guangzhou-demo\imagery/'

    order_info,image_info=load_order_file(order_file)
    image_info=load_rpc_file(image_info)
    tiept_num,tiept_info=load_image_tiept_data(tiept_file)
    tiept_constl=forward_intersec_on_const_level(image_info,tiept_info)
    # tiept_fwd=forward_intersec(image_info,tiept_constl)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from concurrent import futures
import warnings
import time
import seaborn as sns
from multiprocessing import cpu_count

warnings.filterwarnings("ignore")

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

def load_imgtiept_data(tiept_file):
    with open(tiept_file, 'r') as file:
        lines = file.readlines()

    keys=['imgpt_id','object_name','img_id','imgpt_y','imgpt_x']
    tiept_info = {}

    for line in tqdm(lines[1:-1],desc='loading tie points'):
        line = line.strip()

        if line:
            values = line.split('\t')
            for key_value in zip(keys,values):
                if key_value[0] not in tiept_info.keys():
                    if key_value[0]=='object_name':
                         tiept_info[key_value[0]]=[key_value[1]]
                    else:
                        tiept_info[key_value[0]]=[eval(key_value[1])]
                else:
                    if key_value[0]=='object_name':
                        tiept_info[key_value[0]].append(key_value[1])
                    else:
                        tiept_info[key_value[0]].append(eval(key_value[1]))
                            
    for key in tiept_info.keys():
        tiept_info[key]=np.asarray(tiept_info[key])

    # tiept_info=pd.DataFrame(tiept_info)

    tiept_num=len(tiept_info['object_name'])
    tiept_info['objpt_x']=np.zeros(tiept_num)
    tiept_info['objpt_y']=np.zeros(tiept_num)
    tiept_info['objpt_z']=np.zeros(tiept_num)

    return tiept_num,tiept_info

def load_groundtiept_data(tiept_file):
    with open(tiept_file, 'r') as file:
        lines = file.readlines()

    keys=['imgpt_id','object_name','img_id','objpt_x','objpt_y','objpt_z']
    tiept_info = {}

    for line in tqdm(lines[1:-1],desc='loading tie points'):
        line = line.strip()

        if line:
            values = line.split('\t')
            for key_value in zip(keys,values):
                if key_value[0] not in tiept_info.keys():
                    if key_value[0]=='object_name':
                         tiept_info[key_value[0]]=[key_value[1]]
                    else:
                        tiept_info[key_value[0]]=[eval(key_value[1])]
                else:
                    if key_value[0]=='object_name':
                        tiept_info[key_value[0]].append(key_value[1])
                    else:
                        tiept_info[key_value[0]].append(eval(key_value[1]))
                            
    for key in tiept_info.keys():
        tiept_info[key]=np.asarray(tiept_info[key])

    # tiept_info=pd.DataFrame(tiept_info)

    tiept_num=len(tiept_info['object_name'])
    tiept_info['imgpt_x']=np.zeros(tiept_num)
    tiept_info['imgpt_y']=np.zeros(tiept_num)

    return tiept_num,tiept_info

def load_rpc_file(type='rpb'):
    image_num=len(image_info['ImageID'])
    if type=='rpb':
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

    elif type=='txt':
        for i in tqdm(range(image_num),desc=f'loading RPC files'):
            sum_value=[]
            (image_path,image_name)=os.path.split(image_info['ImageFile'][i])
            image_name_split=str.split(image_name,'.')
            rpc_file=os.path.join(image_path,image_name_split[0]+'_rpc.txt')
            with open(rpc_file, 'r') as file:
                lines = file.readlines()

                for j in range(len(lines)):
                    line = lines[j].strip()
                    line_split=str.split(line,':')
                    if j<10:
                        line_split_2nd=str.split(line_split[1],' ')
                        value=eval(line_split_2nd[1])
                        if line_split[0] not in image_info.keys():
                            image_info[line_split[0]]=[value]
                        else:
                            image_info[line_split[0]].append(value)
                    elif 10<=j<30:
                        value=eval(line_split[1])
                        sum_value.append(value)
                        if j==29:
                            if 'lineNumCoef' not in image_info.keys():
                                image_info['lineNumCoef']=[sum_value]
                            else:
                                image_info['lineNumCoef'].append(sum_value)
                            sum_value=[]
                    elif 30<=j<50:
                        value=eval(line_split[1])
                        sum_value.append(value)
                        if j==49:
                            if 'lineDenCoef' not in image_info.keys():
                                image_info['lineDenCoef']=[sum_value]
                            else:
                                image_info['lineDenCoef'].append(sum_value)
                            sum_value=[]
                    elif 50<=j<70:
                        value=eval(line_split[1])
                        sum_value.append(value)
                        if j==69:
                            if 'sampNumCoef' not in image_info.keys():
                                image_info['sampNumCoef']=[sum_value]
                            else:
                                image_info['sampNumCoef'].append(sum_value)
                            sum_value=[]
                    elif 70<=j<90:
                        value=eval(line_split[1])
                        sum_value.append(value)
                        if j==89:
                            if 'sampDenCoef' not in image_info.keys():
                                image_info['sampDenCoef']=[sum_value]
                            else:
                                image_info['sampDenCoef'].append(sum_value)
                            sum_value=[]

    image_info["lineOffset"] = image_info.pop("LINE_OFF")
    image_info["sampOffset"] = image_info.pop("SAMP_OFF")
    image_info["latOffset"] = image_info.pop("LAT_OFF")
    image_info["longOffset"] = image_info.pop("LONG_OFF")
    image_info["heightOffset"] = image_info.pop("HEIGHT_OFF")
    image_info["lineScale"] = image_info.pop("LINE_SCALE")
    image_info["sampScale"] = image_info.pop("SAMP_SCALE")
    image_info["latScale"] = image_info.pop("LAT_SCALE")
    image_info["longScale"] = image_info.pop("LONG_SCALE")
    image_info["heightScale"] = image_info.pop("HEIGHT_SCALE")

    for key in image_info.keys():
        image_info[key]=np.asarray(image_info[key])            

def load_sla_file(sla_file):
    with open(sla_file, 'r') as file:
        lines = file.readlines()

    slapt_num=eval(lines[0])

    for line in tqdm(lines[1:-1]):
        line = line.strip()

        if line:
            values = line.split('\t')
            keys=tiept_info.keys()
            for i in range(keys):
                tiept_info[keys[i]].append(eval(values[i]))
                            
    return slapt_num

def forward_intersec_single_point(obj_name):
    max_iter=100
    eps=1e-5

    indices=np.where(tiept_info['object_name']==obj_name)[0]

    for idx in indices:
    
        flag=True
        
        image_id=tiept_info['img_id'][idx].astype('int')

        lon=image_info['longOffset'][image_id]
        lat=image_info['latOffset'][image_id]
        hei=image_info['heightOffset'][image_id]

        if image_id in image_info['ImageID']:
            X=np.zeros((3))
            A=np.zeros((2,2))
            f=np.zeros((2))
            P=np.eye(2)

            iter_count=0

            while flag:
                tqdm.write(f'forward intersection(single point): the {iter_count+1}th iteration of tiepoint {obj_name}')
                
                L=(lon-image_info['longOffset'][image_id])/image_info['longScale'][image_id]
                B=(lat-image_info['latOffset'][image_id])/image_info['latScale'][image_id]
                H=(hei-image_info['heightOffset'][image_id])/image_info['heightScale'][image_id]

                BL=B*L;BH=B*H;LH=L*H;BB=B*B;LL=L*L;HH=H*H;BLH=B*L*H;BBL=B*B*L;BBH=B*B*H;LLB=L*L*B;LLH=L*L*H;HHB=H*H*B;HHL=H*H*L;BBB=B*B*B;LLL=L*L*L;HHH=H*H*H
                LNUM=image_info['lineNumCoef'][image_id][0]+image_info['lineNumCoef'][image_id][1]*L+image_info['lineNumCoef'][image_id][2]*B+image_info['lineNumCoef'][image_id][3]*H\
                    +image_info['lineNumCoef'][image_id][4]*BL+image_info['lineNumCoef'][image_id][5]*LH+image_info['lineNumCoef'][image_id][6]*BH\
                    +image_info['lineNumCoef'][image_id][7]*LL+image_info['lineNumCoef'][image_id][8]*BB+image_info['lineNumCoef'][image_id][9]*HH\
                    +image_info['lineNumCoef'][image_id][10]*BLH+image_info['lineNumCoef'][image_id][11]*LLL+image_info['lineNumCoef'][image_id][12]*BBL\
                    +image_info['lineNumCoef'][image_id][13]*HHL+image_info['lineNumCoef'][image_id][14]*LLB+image_info['lineNumCoef'][image_id][15]*BBB\
                    +image_info['lineNumCoef'][image_id][16]*HHB+image_info['lineNumCoef'][image_id][17]*LLH+image_info['lineNumCoef'][image_id][18]*BBH\
                    +image_info['lineNumCoef'][image_id][19]*HHH
            
                LDEN=image_info['lineDenCoef'][image_id][0]+image_info['lineDenCoef'][image_id][1]*L+image_info['lineDenCoef'][image_id][2]*B+image_info['lineDenCoef'][image_id][3]*H\
                        +image_info['lineDenCoef'][image_id][4]*BL+image_info['lineDenCoef'][image_id][5]*LH+image_info['lineDenCoef'][image_id][6]*BH\
                        +image_info['lineDenCoef'][image_id][7]*LL+image_info['lineDenCoef'][image_id][8]*BB+image_info['lineDenCoef'][image_id][9]*HH\
                        +image_info['lineDenCoef'][image_id][10]*BLH+image_info['lineDenCoef'][image_id][11]*LLL+image_info['lineDenCoef'][image_id][12]*BBL\
                        +image_info['lineDenCoef'][image_id][13]*HHL+image_info['lineDenCoef'][image_id][14]*LLB+image_info['lineDenCoef'][image_id][15]*BBB\
                        +image_info['lineDenCoef'][image_id][16]*HHB+image_info['lineDenCoef'][image_id][17]*LLH+image_info['lineDenCoef'][image_id][18]*BBH\
                        +image_info['lineDenCoef'][image_id][19]*HHH
                        
                SNUM=image_info['sampNumCoef'][image_id][0]+image_info['sampNumCoef'][image_id][1]*L+image_info['sampNumCoef'][image_id][2]*B+image_info['sampNumCoef'][image_id][3]*H\
                        +image_info['sampNumCoef'][image_id][4]*BL+image_info['sampNumCoef'][image_id][5]*LH+image_info['sampNumCoef'][image_id][6]*BH\
                        +image_info['sampNumCoef'][image_id][7]*LL+image_info['sampNumCoef'][image_id][8]*BB+image_info['sampNumCoef'][image_id][9]*HH\
                        +image_info['sampNumCoef'][image_id][10]*BLH+image_info['sampNumCoef'][image_id][11]*LLL+image_info['sampNumCoef'][image_id][12]*BBL\
                        +image_info['sampNumCoef'][image_id][13]*HHL+image_info['sampNumCoef'][image_id][14]*LLB+image_info['sampNumCoef'][image_id][15]*BBB\
                        +image_info['sampNumCoef'][image_id][16]*HHB+image_info['sampNumCoef'][image_id][17]*LLH+image_info['sampNumCoef'][image_id][18]*BBH\
                        +image_info['sampNumCoef'][image_id][19]*HHH
                
                SDEN=image_info['sampDenCoef'][image_id][0]+image_info['sampDenCoef'][image_id][1]*L+image_info['sampDenCoef'][image_id][2]*B+image_info['sampDenCoef'][image_id][3]*H\
                        +image_info['sampDenCoef'][image_id][4]*BL+image_info['sampDenCoef'][image_id][5]*LH+image_info['sampDenCoef'][image_id][6]*BH\
                        +image_info['sampDenCoef'][image_id][7]*LL+image_info['sampDenCoef'][image_id][8]*BB+image_info['sampDenCoef'][image_id][9]*HH\
                        +image_info['sampDenCoef'][image_id][10]*BLH+image_info['sampDenCoef'][image_id][11]*LLL+image_info['sampDenCoef'][image_id][12]*BBL\
                        +image_info['sampDenCoef'][image_id][13]*HHL+image_info['sampDenCoef'][image_id][14]*LLB+image_info['sampDenCoef'][image_id][15]*BBB\
                        +image_info['sampDenCoef'][image_id][16]*HHB+image_info['sampDenCoef'][image_id][17]*LLH+image_info['sampDenCoef'][image_id][18]*BBH\
                        +image_info['sampDenCoef'][image_id][19]*HHH
                
                dLNUMB=image_info['lineNumCoef'][image_id][2]+image_info['lineNumCoef'][image_id][4]*L+image_info['lineNumCoef'][image_id][6]*H+2*image_info['lineNumCoef'][image_id][8]*B+image_info['lineNumCoef'][image_id][10]*LH+2*image_info['lineNumCoef'][image_id][12]*BL+\
                        image_info['lineNumCoef'][image_id][14]*LL+3*image_info['lineNumCoef'][image_id][15]*BB+image_info['lineNumCoef'][image_id][16]*HH+2*image_info['lineNumCoef'][image_id][18]*BH
                dLNUML=image_info['lineNumCoef'][image_id][1]+image_info['lineNumCoef'][image_id][4]*B+image_info['lineNumCoef'][image_id][5]*H+2*image_info['lineNumCoef'][image_id][7]*L+image_info['lineNumCoef'][image_id][10]*BH\
                    +3*image_info['lineNumCoef'][image_id][11]*LL+image_info['lineNumCoef'][image_id][12]*BB+image_info['lineNumCoef'][image_id][13]*HH+2*image_info['lineNumCoef'][image_id][14]*BL+2*image_info['lineNumCoef'][image_id][17]*H*L
                dLNUMH=image_info['lineNumCoef'][image_id][3]+image_info['lineNumCoef'][image_id][5]*L+image_info['lineNumCoef'][image_id][6]*B+2*image_info['lineNumCoef'][image_id][9]*H\
                    +image_info['lineNumCoef'][image_id][10]*BL+2*image_info['lineNumCoef'][image_id][13]*L*H+2*image_info['lineNumCoef'][image_id][16]*B*H+image_info['lineNumCoef'][image_id][17]*LL+image_info['lineNumCoef'][image_id][18]*BB+3*image_info['lineNumCoef'][image_id][19]*HH

                dLDENB=image_info['lineDenCoef'][image_id][2]+image_info['lineDenCoef'][image_id][4]*L+image_info['lineDenCoef'][image_id][6]*H+2*image_info['lineDenCoef'][image_id][8]*B+image_info['lineDenCoef'][image_id][10]*LH+2*image_info['lineDenCoef'][image_id][12]*BL+\
                    image_info['lineDenCoef'][image_id][14]*LL+3*image_info['lineDenCoef'][image_id][15]*BB+image_info['lineDenCoef'][image_id][16]*HH+2*image_info['lineDenCoef'][image_id][18]*BH
                dLDENL=image_info['lineDenCoef'][image_id][1]+image_info['lineDenCoef'][image_id][4]*B+image_info['lineDenCoef'][image_id][5]*H+2*image_info['lineDenCoef'][image_id][7]*L+image_info['lineDenCoef'][image_id][10]*BH\
                    +3*image_info['lineDenCoef'][image_id][11]*LL+image_info['lineDenCoef'][image_id][12]*BB+image_info['lineDenCoef'][image_id][13]*HH+2*image_info['lineDenCoef'][image_id][14]*BL+2*image_info['lineDenCoef'][image_id][17]*H*L
                dLDENH=image_info['lineDenCoef'][image_id][3]+image_info['lineDenCoef'][image_id][5]*L+image_info['lineDenCoef'][image_id][6]*B+2*image_info['lineDenCoef'][image_id][9]*H\
                    +image_info['lineDenCoef'][image_id][10]*BL+2*image_info['lineDenCoef'][image_id][13]*L*H+2*image_info['lineDenCoef'][image_id][16]*B*H+image_info['lineDenCoef'][image_id][17]*LL+image_info['lineDenCoef'][image_id][18]*BB+3*image_info['lineDenCoef'][image_id][19]*HH

                dSNUMB=image_info['sampNumCoef'][image_id][2]+image_info['sampNumCoef'][image_id][4]*L+image_info['sampNumCoef'][image_id][6]*H+2*image_info['sampNumCoef'][image_id][8]*B+image_info['sampNumCoef'][image_id][10]*LH+2*image_info['sampNumCoef'][image_id][12]*BL+\
                    image_info['sampNumCoef'][image_id][14]*LL+3*image_info['sampNumCoef'][image_id][15]*BB+image_info['sampNumCoef'][image_id][16]*HH+2*image_info['sampNumCoef'][image_id][18]*BH
                dSNUML=image_info['sampNumCoef'][image_id][1]+image_info['sampNumCoef'][image_id][4]*B+image_info['sampNumCoef'][image_id][5]*H+2*image_info['sampNumCoef'][image_id][7]*L+image_info['sampNumCoef'][image_id][10]*BH\
                    +3*image_info['sampNumCoef'][image_id][11]*LL+image_info['sampNumCoef'][image_id][12]*BB+image_info['sampNumCoef'][image_id][13]*HH+2*image_info['sampNumCoef'][image_id][14]*BL+2*image_info['sampNumCoef'][image_id][17]*H*L
                dSNUMH=image_info['sampNumCoef'][image_id][3]+image_info['sampNumCoef'][image_id][5]*L+image_info['sampNumCoef'][image_id][6]*B+2*image_info['sampNumCoef'][image_id][9]*H\
                    +image_info['sampNumCoef'][image_id][10]*BL+2*image_info['sampNumCoef'][image_id][13]*L*H+2*image_info['sampNumCoef'][image_id][16]*B*H+image_info['sampNumCoef'][image_id][17]*LL+image_info['sampNumCoef'][image_id][18]*BB+3*image_info['sampNumCoef'][image_id][19]*HH

                dSDENB=image_info['sampDenCoef'][image_id][2]+image_info['sampDenCoef'][image_id][4]*L+image_info['sampDenCoef'][image_id][6]*H+2*image_info['sampDenCoef'][image_id][8]*B+image_info['sampDenCoef'][image_id][10]*LH+2*image_info['sampDenCoef'][image_id][12]*BL+\
                    image_info['sampDenCoef'][image_id][14]*LL+3*image_info['sampDenCoef'][image_id][15]*BB+image_info['sampDenCoef'][image_id][16]*HH+2*image_info['sampDenCoef'][image_id][18]*BH
                dSDENL=image_info['sampDenCoef'][image_id][1]+image_info['sampDenCoef'][image_id][4]*B+image_info['sampDenCoef'][image_id][5]*H+2*image_info['sampDenCoef'][image_id][7]*L+image_info['sampDenCoef'][image_id][10]*BH\
                    +3*image_info['sampDenCoef'][image_id][11]*LL+image_info['sampDenCoef'][image_id][12]*BB+image_info['sampDenCoef'][image_id][13]*HH+2*image_info['sampDenCoef'][image_id][14]*BL+2*image_info['sampDenCoef'][image_id][17]*H*L
                dSDENH=image_info['sampDenCoef'][image_id][3]+image_info['sampDenCoef'][image_id][5]*L+image_info['sampDenCoef'][image_id][6]*B+2*image_info['sampDenCoef'][image_id][9]*H\
                    +image_info['sampDenCoef'][image_id][10]*BL+2*image_info['sampDenCoef'][image_id][13]*L*H+2*image_info['sampDenCoef'][image_id][16]*B*H+image_info['sampDenCoef'][image_id][17]*LL+image_info['sampDenCoef'][image_id][18]*BB+3*image_info['sampDenCoef'][image_id][19]*HH

                dLB=(dLNUMB*LDEN-LNUM*dLDENB)/(LDEN*LDEN)
                dLL=(dLNUML*LDEN-LNUM*dLDENL)/(LDEN*LDEN)
                dLH=(dLNUMH*LDEN-LNUM*dLDENH)/(LDEN*LDEN)

                dSB=(dSNUMB*SDEN-SNUM*dSDENB)/(SDEN*SDEN)
                dSL=(dSNUML*SDEN-SNUM*dSDENL)/(SDEN*SDEN)
                dSH=(dSNUMH*SDEN-SNUM*dSDENH)/(SDEN*SDEN)

                dLlat=dLB/image_info['latScale'][image_id];dLlon=dLL/image_info['longScale'][image_id];dLhei=dLH/image_info['heightScale'][image_id]
                dSlat=dSB/image_info['latScale'][image_id];dSlon=dSL/image_info['longScale'][image_id];dShei=dSH/image_info['heightScale'][image_id]

                l=np.asarray(tiept_info['imgpt_x'][idx])
                s=np.asarray(tiept_info['imgpt_y'][idx])

                A=np.asarray([[dLlon*image_info['lineScale'][image_id],dLlat*image_info['lineScale'][image_id]],\
                                                                [dSlon*image_info['sampScale'][image_id],dSlat*image_info['sampScale'][image_id]]])
                fl_0=(LNUM/LDEN)*image_info['lineScale'][image_id]+image_info['lineOffset'][image_id]
                fs_0=(SNUM/SDEN)*image_info['sampScale'][image_id]+image_info['sampOffset'][image_id]
                                
                f[0]=l-fl_0
                f[1]=s-fs_0
                f=f

                coeff=A.T@P@A
                Q_xx=np.linalg.inv(coeff)
                adj_esti=Q_xx@A.T@P@f

                X=adj_esti

                iter_count+=1

                lon+=X[0]
                lat+=X[1]

                if np.fabs(np.max(adj_esti))<=eps or iter_count>=max_iter:
                    flag=False

        tiept_info['objpt_x'][idx]=lon
        tiept_info['objpt_y'][idx]=lat
        tiept_info['objpt_z'][idx]=hei

def forward_intersec_multiple_points(obj_name):
    max_iter=100
    eps=1e-5
    indices=np.where(tiept_info['object_name']==obj_name)[0]

    flag=True
    tiept_num=len(indices)
    image_id=tiept_info['img_id'][indices].astype('int')

    X=np.zeros((3*tiept_num))
    A=np.zeros((2*tiept_num,3))
    f=np.zeros((2*tiept_num))
    P=np.eye(2*tiept_num)

    iter_count=0

    lon=tiept_info['objpt_x'][indices].astype('float32')
    lat=tiept_info['objpt_y'][indices].astype('float32')
    hei=tiept_info['objpt_z'][indices].astype('float32')

    while flag:
        # tqdm.write(f'f orward intersection(multiple points): the {iter_count+1}th iteration of tiepoint {obj_name}')
        
        L=(lon-image_info['longOffset'][image_id])/image_info['longScale'][image_id]
        B=(lat-image_info['latOffset'][image_id])/image_info['latScale'][image_id]
        H=(hei-image_info['heightOffset'][image_id])/image_info['heightScale'][image_id]

        BL=B*L;BH=B*H;LH=L*H;BB=B*B;LL=L*L;HH=H*H;BLH=B*L*H;BBL=B*B*L;BBH=B*B*H;LLB=L*L*B;LLH=L*L*H;HHB=H*H*B;HHL=H*H*L;BBB=B*B*B;LLL=L*L*L;HHH=H*H*H
        LNUM=image_info['lineNumCoef'][image_id][:,0]+image_info['lineNumCoef'][image_id][:,1]*L+image_info['lineNumCoef'][image_id][:,2]*B+image_info['lineNumCoef'][image_id][:,3]*H\
                +image_info['lineNumCoef'][image_id][:,4]*BL+image_info['lineNumCoef'][image_id][:,5]*LH+image_info['lineNumCoef'][image_id][:,6]*BH\
                +image_info['lineNumCoef'][image_id][:,7]*LL+image_info['lineNumCoef'][image_id][:,8]*BB+image_info['lineNumCoef'][image_id][:,9]*HH\
                +image_info['lineNumCoef'][image_id][:,10]*BLH+image_info['lineNumCoef'][image_id][:,11]*LLL+image_info['lineNumCoef'][image_id][:,12]*BBL\
                +image_info['lineNumCoef'][image_id][:,13]*HHL+image_info['lineNumCoef'][image_id][:,14]*LLB+image_info['lineNumCoef'][image_id][:,15]*BBB\
                +image_info['lineNumCoef'][image_id][:,16]*HHB+image_info['lineNumCoef'][image_id][:,17]*LLH+image_info['lineNumCoef'][image_id][:,18]*BBH\
                +image_info['lineNumCoef'][image_id][:,19]*HHH
                
        LDEN=image_info['lineDenCoef'][image_id][:,0]+image_info['lineDenCoef'][image_id][:,1]*L+image_info['lineDenCoef'][image_id][:,2]*B+image_info['lineDenCoef'][image_id][:,3]*H\
                +image_info['lineDenCoef'][image_id][:,4]*BL+image_info['lineDenCoef'][image_id][:,5]*LH+image_info['lineDenCoef'][image_id][:,6]*BH\
                +image_info['lineDenCoef'][image_id][:,7]*LL+image_info['lineDenCoef'][image_id][:,8]*BB+image_info['lineDenCoef'][image_id][:,9]*HH\
                +image_info['lineDenCoef'][image_id][:,10]*BLH+image_info['lineDenCoef'][image_id][:,11]*LLL+image_info['lineDenCoef'][image_id][:,12]*BBL\
                +image_info['lineDenCoef'][image_id][:,13]*HHL+image_info['lineDenCoef'][image_id][:,14]*LLB+image_info['lineDenCoef'][image_id][:,15]*BBB\
                +image_info['lineDenCoef'][image_id][:,16]*HHB+image_info['lineDenCoef'][image_id][:,17]*LLH+image_info['lineDenCoef'][image_id][:,18]*BBH\
                +image_info['lineDenCoef'][image_id][:,19]*HHH
                
        SNUM=image_info['sampNumCoef'][image_id][:,0]+image_info['sampNumCoef'][image_id][:,1]*L+image_info['sampNumCoef'][image_id][:,2]*B+image_info['sampNumCoef'][image_id][:,3]*H\
                +image_info['sampNumCoef'][image_id][:,4]*BL+image_info['sampNumCoef'][image_id][:,5]*LH+image_info['sampNumCoef'][image_id][:,6]*BH\
                +image_info['sampNumCoef'][image_id][:,7]*LL+image_info['sampNumCoef'][image_id][:,8]*BB+image_info['sampNumCoef'][image_id][:,9]*HH\
                +image_info['sampNumCoef'][image_id][:,10]*BLH+image_info['sampNumCoef'][image_id][:,11]*LLL+image_info['sampNumCoef'][image_id][:,12]*BBL\
                +image_info['sampNumCoef'][image_id][:,13]*HHL+image_info['sampNumCoef'][image_id][:,14]*LLB+image_info['sampNumCoef'][image_id][:,15]*BBB\
                +image_info['sampNumCoef'][image_id][:,16]*HHB+image_info['sampNumCoef'][image_id][:,17]*LLH+image_info['sampNumCoef'][image_id][:,18]*BBH\
                +image_info['sampNumCoef'][image_id][:,19]*HHH
        
        SDEN=image_info['sampDenCoef'][image_id][:,0]+image_info['sampDenCoef'][image_id][:,1]*L+image_info['sampDenCoef'][image_id][:,2]*B+image_info['sampDenCoef'][image_id][:,3]*H\
                +image_info['sampDenCoef'][image_id][:,4]*BL+image_info['sampDenCoef'][image_id][:,5]*LH+image_info['sampDenCoef'][image_id][:,6]*BH\
                +image_info['sampDenCoef'][image_id][:,7]*LL+image_info['sampDenCoef'][image_id][:,8]*BB+image_info['sampDenCoef'][image_id][:,9]*HH\
                +image_info['sampDenCoef'][image_id][:,10]*BLH+image_info['sampDenCoef'][image_id][:,11]*LLL+image_info['sampDenCoef'][image_id][:,12]*BBL\
                +image_info['sampDenCoef'][image_id][:,13]*HHL+image_info['sampDenCoef'][image_id][:,14]*LLB+image_info['sampDenCoef'][image_id][:,15]*BBB\
                +image_info['sampDenCoef'][image_id][:,16]*HHB+image_info['sampDenCoef'][image_id][:,17]*LLH+image_info['sampDenCoef'][image_id][:,18]*BBH\
                +image_info['sampDenCoef'][image_id][:,19]*HHH
        
        dLNUMB=image_info['lineNumCoef'][image_id][:,2]+image_info['lineNumCoef'][image_id][:,4]*L+image_info['lineNumCoef'][image_id][:,6]*H+2*image_info['lineNumCoef'][image_id][:,8]*B+image_info['lineNumCoef'][image_id][:,10]*LH+2*image_info['lineNumCoef'][image_id][:,12]*BL+\
            image_info['lineNumCoef'][image_id][:,14]*LL+3*image_info['lineNumCoef'][image_id][:,15]*BB+image_info['lineNumCoef'][image_id][:,16]*HH+2*image_info['lineNumCoef'][image_id][:,18]*BH
        dLNUML=image_info['lineNumCoef'][image_id][:,1]+image_info['lineNumCoef'][image_id][:,4]*B+image_info['lineNumCoef'][image_id][:,5]*H+2*image_info['lineNumCoef'][image_id][:,7]*L+image_info['lineNumCoef'][image_id][:,10]*BH\
            +3*image_info['lineNumCoef'][image_id][:,11]*LL+image_info['lineNumCoef'][image_id][:,12]*BB+image_info['lineNumCoef'][image_id][:,13]*HH+2*image_info['lineNumCoef'][image_id][:,14]*BL+2*image_info['lineNumCoef'][image_id][:,17]*H*L
        dLNUMH=image_info['lineNumCoef'][image_id][:,3]+image_info['lineNumCoef'][image_id][:,5]*L+image_info['lineNumCoef'][image_id][:,6]*B+2*image_info['lineNumCoef'][image_id][:,9]*H\
            +image_info['lineNumCoef'][image_id][:,10]*BL+2*image_info['lineNumCoef'][image_id][:,13]*L*H+2*image_info['lineNumCoef'][image_id][:,16]*B*H+image_info['lineNumCoef'][image_id][:,17]*LL+image_info['lineNumCoef'][image_id][:,18]*BB+3*image_info['lineNumCoef'][image_id][:,19]*HH

        dLDENB=image_info['lineDenCoef'][image_id][:,2]+image_info['lineDenCoef'][image_id][:,4]*L+image_info['lineDenCoef'][image_id][:,6]*H+2*image_info['lineDenCoef'][image_id][:,8]*B+image_info['lineDenCoef'][image_id][:,10]*LH+2*image_info['lineDenCoef'][image_id][:,12]*BL+\
            image_info['lineDenCoef'][image_id][:,14]*LL+3*image_info['lineDenCoef'][image_id][:,15]*BB+image_info['lineDenCoef'][image_id][:,16]*HH+2*image_info['lineDenCoef'][image_id][:,18]*BH
        dLDENL=image_info['lineDenCoef'][image_id][:,1]+image_info['lineDenCoef'][image_id][:,4]*B+image_info['lineDenCoef'][image_id][:,5]*H+2*image_info['lineDenCoef'][image_id][:,7]*L+image_info['lineDenCoef'][image_id][:,10]*BH\
            +3*image_info['lineDenCoef'][image_id][:,11]*LL+image_info['lineDenCoef'][image_id][:,12]*BB+image_info['lineDenCoef'][image_id][:,13]*HH+2*image_info['lineDenCoef'][image_id][:,14]*BL+2*image_info['lineDenCoef'][image_id][:,17]*H*L
        dLDENH=image_info['lineDenCoef'][image_id][:,3]+image_info['lineDenCoef'][image_id][:,5]*L+image_info['lineDenCoef'][image_id][:,6]*B+2*image_info['lineDenCoef'][image_id][:,9]*H\
            +image_info['lineDenCoef'][image_id][:,10]*BL+2*image_info['lineDenCoef'][image_id][:,13]*L*H+2*image_info['lineDenCoef'][image_id][:,16]*B*H+image_info['lineDenCoef'][image_id][:,17]*LL+image_info['lineDenCoef'][image_id][:,18]*BB+3*image_info['lineDenCoef'][image_id][:,19]*HH

        dSNUMB=image_info['sampNumCoef'][image_id][:,2]+image_info['sampNumCoef'][image_id][:,4]*L+image_info['sampNumCoef'][image_id][:,6]*H+2*image_info['sampNumCoef'][image_id][:,8]*B+image_info['sampNumCoef'][image_id][:,10]*LH+2*image_info['sampNumCoef'][image_id][:,12]*BL+\
            image_info['sampNumCoef'][image_id][:,14]*LL+3*image_info['sampNumCoef'][image_id][:,15]*BB+image_info['sampNumCoef'][image_id][:,16]*HH+2*image_info['sampNumCoef'][image_id][:,18]*BH
        dSNUML=image_info['sampNumCoef'][image_id][:,1]+image_info['sampNumCoef'][image_id][:,4]*B+image_info['sampNumCoef'][image_id][:,5]*H+2*image_info['sampNumCoef'][image_id][:,7]*L+image_info['sampNumCoef'][image_id][:,10]*BH\
            +3*image_info['sampNumCoef'][image_id][:,11]*LL+image_info['sampNumCoef'][image_id][:,12]*BB+image_info['sampNumCoef'][image_id][:,13]*HH+2*image_info['sampNumCoef'][image_id][:,14]*BL+2*image_info['sampNumCoef'][image_id][:,17]*H*L
        dSNUMH=image_info['sampNumCoef'][image_id][:,3]+image_info['sampNumCoef'][image_id][:,5]*L+image_info['sampNumCoef'][image_id][:,6]*B+2*image_info['sampNumCoef'][image_id][:,9]*H\
            +image_info['sampNumCoef'][image_id][:,10]*BL+2*image_info['sampNumCoef'][image_id][:,13]*L*H+2*image_info['sampNumCoef'][image_id][:,16]*B*H+image_info['sampNumCoef'][image_id][:,17]*LL+image_info['sampNumCoef'][image_id][:,18]*BB+3*image_info['sampNumCoef'][image_id][:,19]*HH

        dSDENB=image_info['sampDenCoef'][image_id][:,2]+image_info['sampDenCoef'][image_id][:,4]*L+image_info['sampDenCoef'][image_id][:,6]*H+2*image_info['sampDenCoef'][image_id][:,8]*B+image_info['sampDenCoef'][image_id][:,10]*LH+2*image_info['sampDenCoef'][image_id][:,12]*BL+\
            image_info['sampDenCoef'][image_id][:,14]*LL+3*image_info['sampDenCoef'][image_id][:,15]*BB+image_info['sampDenCoef'][image_id][:,16]*HH+2*image_info['sampDenCoef'][image_id][:,18]*BH
        dSDENL=image_info['sampDenCoef'][image_id][:,1]+image_info['sampDenCoef'][image_id][:,4]*B+image_info['sampDenCoef'][image_id][:,5]*H+2*image_info['sampDenCoef'][image_id][:,7]*L+image_info['sampDenCoef'][image_id][:,10]*BH\
            +3*image_info['sampDenCoef'][image_id][:,11]*LL+image_info['sampDenCoef'][image_id][:,12]*BB+image_info['sampDenCoef'][image_id][:,13]*HH+2*image_info['sampDenCoef'][image_id][:,14]*BL+2*image_info['sampDenCoef'][image_id][:,17]*H*L
        dSDENH=image_info['sampDenCoef'][image_id][:,3]+image_info['sampDenCoef'][image_id][:,5]*L+image_info['sampDenCoef'][image_id][:,6]*B+2*image_info['sampDenCoef'][image_id][:,9]*H\
            +image_info['sampDenCoef'][image_id][:,10]*BL+2*image_info['sampDenCoef'][image_id][:,13]*L*H+2*image_info['sampDenCoef'][image_id][:,16]*B*H+image_info['sampDenCoef'][image_id][:,17]*LL+image_info['sampDenCoef'][image_id][:,18]*BB+3*image_info['sampDenCoef'][image_id][:,19]*HH

        dLB=(dLNUMB*LDEN-LNUM*dLDENB)/(LDEN*LDEN)
        dLL=(dLNUML*LDEN-LNUM*dLDENL)/(LDEN*LDEN)
        dLH=(dLNUMH*LDEN-LNUM*dLDENH)/(LDEN*LDEN)

        dSB=(dSNUMB*SDEN-SNUM*dSDENB)/(SDEN*SDEN)
        dSL=(dSNUML*SDEN-SNUM*dSDENL)/(SDEN*SDEN)
        dSH=(dSNUMH*SDEN-SNUM*dSDENH)/(SDEN*SDEN)

        dLlat=dLB/image_info['latScale'][image_id];dLlon=dLL/image_info['longScale'][image_id];dLhei=dLH/image_info['heightScale'][image_id]
        dSlat=dSB/image_info['latScale'][image_id];dSlon=dSL/image_info['longScale'][image_id];dShei=dSH/image_info['heightScale'][image_id]

        l=np.asarray(tiept_info['imgpt_x'][indices])
        s=np.asarray(tiept_info['imgpt_y'][indices])

        for i in range(tiept_num):
            A[i*2:(i+1)*2,:]=np.asarray([[dLlon[i]*image_info['lineScale'][image_id[i]],dLlat[i]*image_info['lineScale'][image_id[i]],dLhei[i]*image_info['lineScale'][image_id[i]]],\
                                                            [dSlon[i]*image_info['sampScale'][image_id[i]],dSlat[i]*image_info['sampScale'][image_id[i]],dShei[i]*image_info['sampScale'][image_id[i]]]])
            fl_0=(LNUM[i]/LDEN[i])*image_info['lineScale'][image_id[i]]+image_info['lineOffset'][image_id[i]]
            fs_0=(SNUM[i]/SDEN[i])*image_info['sampScale'][image_id[i]]+image_info['sampOffset'][image_id[i]]
                            
            f[i*2]=l[i]-fl_0
            f[i*2+1]=s[i]-fs_0

        coeff=A.T@P@A
        Q_xx=np.linalg.inv(coeff)
        adj_esti=Q_xx@A.T@P@f

        X=adj_esti

        iter_count+=1

        lon+=X[0]
        lat+=X[1]
        hei+=X[2]

        if np.fabs(np.max(adj_esti))<=eps or iter_count>=max_iter:
            flag=False

    tiept_info['objpt_x'][indices]=lon
    tiept_info['objpt_y'][indices]=lat
    tiept_info['objpt_z'][indices]=hei
    print('idx',indices,'lon',lon,'lat',lat,'hei',hei)

def forward_intersec_on_const_level():
    objname_set,counts=np.unique(tiept_info['object_name'],return_counts=True)
    duplicate_indices=np.where(counts>1)[0]
    valid_objnames=objname_set[duplicate_indices]
      
    tiept_idx = np.isin(tiept_info['object_name'],valid_objnames)

    for key in tiept_info.keys():
        tiept_info[key]=tiept_info[key][tiept_idx]

    with futures.ThreadPoolExecutor(max_workers=max_core) as texecutor:
        texecutor.map(forward_intersec_single_point,valid_objnames)

def forward_intersec():
    objname_set,counts=np.unique(tiept_info['object_name'],return_counts=True)
    duplicate_indices=np.where(counts>1)[0]
    valid_objname=objname_set[duplicate_indices]
      
    tiept_idx = np.isin(tiept_info['object_name'],valid_objname)

    for key in tiept_info.keys():
        tiept_info[key]=tiept_info[key][tiept_idx]

    with futures.ThreadPoolExecutor(max_workers=max_core) as texecutor:
        texecutor.map(forward_intersec_multiple_points,valid_objname)

def refine_para_compute(refine_model=None):
    img_num=len(image_info['ImageID'])

    if refine_model=='translation':
        refine_num=2
        image_info['refine_paras']=np.zeros((img_num,refine_num))
    elif refine_model=='affine':
        refine_num=6
        image_info['refine_paras']=np.zeros((img_num,refine_num))

    tiept_curr_img=dict(key=['img_id','imgpt_x','img_pt_y','object_name','objpt_x','objpt_y','objpt_z'])

    for id in image_info['ImageID']:
        indices=np.where(tiept_info['img_id']==id)[0]
        tiept_num=len(indices)
        # tiept_curr_img=tiept_info.loc[indices]
        # tiept_curr_img=tiept_info[indices]
        for key in tiept_info.keys():
            tiept_curr_img[key]=tiept_info[key][indices]

        L=tiept_curr_img['objpt_x'].astype('float32')
        B=tiept_curr_img['objpt_y'].astype('float32')
        H=tiept_curr_img['objpt_z'].astype('float32')

        L=(L-image_info['longOffset'][id])/image_info['longScale'][id]
        B=(B-image_info['latOffset'][id])/image_info['latScale'][id]
        H=(H-image_info['heightOffset'][id])/image_info['heightScale'][id]

        BL=B*L;BH=B*H;LH=L*H;BB=B*B;LL=L*L;HH=H*H;BLH=B*L*H;BBL=B*B*L;BBH=B*B*H;LLB=L*L*B;LLH=L*L*H;HHB=H*H*B;HHL=H*H*L;BBB=B*B*B;LLL=L*L*L;HHH=H*H*H
        LNUM=image_info['lineNumCoef'][id][0]+image_info['lineNumCoef'][id][1]*L+image_info['lineNumCoef'][id][2]*B+image_info['lineNumCoef'][id][3]*H\
                        +image_info['lineNumCoef'][id][4]*BL+image_info['lineNumCoef'][id][5]*LH+image_info['lineNumCoef'][id][6]*BH\
                        +image_info['lineNumCoef'][id][7]*LL+image_info['lineNumCoef'][id][8]*BB+image_info['lineNumCoef'][id][9]*HH\
                        +image_info['lineNumCoef'][id][10]*BLH+image_info['lineNumCoef'][id][11]*LLL+image_info['lineNumCoef'][id][12]*BBL\
                        +image_info['lineNumCoef'][id][13]*HHL+image_info['lineNumCoef'][id][14]*LLB+image_info['lineNumCoef'][id][15]*BBB\
                        +image_info['lineNumCoef'][id][16]*HHB+image_info['lineNumCoef'][id][17]*LLH+image_info['lineNumCoef'][id][18]*BBH\
                        +image_info['lineNumCoef'][id][19]*HHH
                
        LDEN=image_info['lineDenCoef'][id][0]+image_info['lineDenCoef'][id][1]*L+image_info['lineDenCoef'][id][2]*B+image_info['lineDenCoef'][id][3]*H\
                +image_info['lineDenCoef'][id][4]*BL+image_info['lineDenCoef'][id][5]*LH+image_info['lineDenCoef'][id][6]*BH\
                +image_info['lineDenCoef'][id][7]*LL+image_info['lineDenCoef'][id][8]*BB+image_info['lineDenCoef'][id][9]*HH\
                +image_info['lineDenCoef'][id][10]*BLH+image_info['lineDenCoef'][id][11]*LLL+image_info['lineDenCoef'][id][12]*BBL\
                +image_info['lineDenCoef'][id][13]*HHL+image_info['lineDenCoef'][id][14]*LLB+image_info['lineDenCoef'][id][15]*BBB\
                +image_info['lineDenCoef'][id][16]*HHB+image_info['lineDenCoef'][id][17]*LLH+image_info['lineDenCoef'][id][18]*BBH\
                +image_info['lineDenCoef'][id][19]*HHH
                
        SNUM=image_info['sampNumCoef'][id][0]+image_info['sampNumCoef'][id][1]*L+image_info['sampNumCoef'][id][2]*B+image_info['sampNumCoef'][id][3]*H\
                +image_info['sampNumCoef'][id][4]*BL+image_info['sampNumCoef'][id][5]*LH+image_info['sampNumCoef'][id][6]*BH\
                +image_info['sampNumCoef'][id][7]*LL+image_info['sampNumCoef'][id][8]*BB+image_info['sampNumCoef'][id][9]*HH\
                +image_info['sampNumCoef'][id][10]*BLH+image_info['sampNumCoef'][id][11]*LLL+image_info['sampNumCoef'][id][12]*BBL\
                +image_info['sampNumCoef'][id][13]*HHL+image_info['sampNumCoef'][id][14]*LLB+image_info['sampNumCoef'][id][15]*BBB\
                +image_info['sampNumCoef'][id][16]*HHB+image_info['sampNumCoef'][id][17]*LLH+image_info['sampNumCoef'][id][18]*BBH\
                +image_info['sampNumCoef'][id][19]*HHH
        
        SDEN=image_info['sampDenCoef'][id][0]+image_info['sampDenCoef'][id][1]*L+image_info['sampDenCoef'][id][2]*B+image_info['sampDenCoef'][id][3]*H\
                +image_info['sampDenCoef'][id][4]*BL+image_info['sampDenCoef'][id][5]*LH+image_info['sampDenCoef'][id][6]*BH\
                +image_info['sampDenCoef'][id][7]*LL+image_info['sampDenCoef'][id][8]*BB+image_info['sampDenCoef'][id][9]*HH\
                +image_info['sampDenCoef'][id][10]*BLH+image_info['sampDenCoef'][id][11]*LLL+image_info['sampDenCoef'][id][12]*BBL\
                +image_info['sampDenCoef'][id][13]*HHL+image_info['sampDenCoef'][id][14]*LLB+image_info['sampDenCoef'][id][15]*BBB\
                +image_info['sampDenCoef'][id][16]*HHB+image_info['sampDenCoef'][id][17]*LLH+image_info['sampDenCoef'][id][18]*BBH\
                +image_info['sampDenCoef'][id][19]*HHH
        
        A=np.zeros((tiept_num*2,refine_num))
        P=np.eye(tiept_num*2)
        f=np.zeros((tiept_num*2))        
        refine_paras=np.zeros((refine_num))
        X=np.zeros((refine_num))

        eps=1e-5
        max_iter=100

        flag=True

        iter_count=0

        while flag:
            single_num=int(refine_num/2)
            if refine_model=='translation':
                A[0:tiept_num*2:2,0:single_num]=-1
                A[1:tiept_num*2+1:2,single_num:A.shape[1]]=-1
                f[0:tiept_num*2:2]=tiept_curr_img['imgpt_x']+refine_paras[0]-((LNUM/LDEN)*image_info['lineScale'][id]+image_info['lineOffset'][id])
                f[1:tiept_num*2+1:2]=tiept_curr_img['imgpt_y']+refine_paras[1]-((SNUM/SDEN)*image_info['sampScale'][id]+image_info['sampOffset'][id])
            elif refine_model=='affine':
                A[0:tiept_num*2:2,0:single_num]=list(zip(-np.ones(len(-tiept_curr_img['imgpt_x'])),-tiept_curr_img['imgpt_x'],-tiept_curr_img['imgpt_y']))
                A[1:tiept_num*2+1:2,single_num:A.shape[1]]=list(zip(-np.ones(len(-tiept_curr_img['imgpt_x'])),-tiept_curr_img['imgpt_x'],-tiept_curr_img['imgpt_y']))
                f[0:tiept_num*2:2]=tiept_curr_img['imgpt_x']+refine_paras[0]+refine_paras[1]*tiept_curr_img['imgpt_x']+refine_paras[2]*tiept_curr_img['imgpt_y']\
                    -((LNUM/LDEN)*image_info['lineScale'][id]+image_info['lineOffset'][id])
                f[1:tiept_num*2+1:2]=tiept_curr_img['imgpt_y']+refine_paras[3]+refine_paras[4]*tiept_curr_img['imgpt_x']+refine_paras[5]*tiept_curr_img['imgpt_y']\
                    -((SNUM/SDEN)*image_info['sampScale'][id]+image_info['sampOffset'][id])

            coeff=A.T@P@A
            Q_xx=np.linalg.inv(coeff)
            X=Q_xx@A.T@P@f

            iter_count+=1

            refine_paras+=X

            if np.max(np.fabs(X))<=eps or iter_count>=max_iter:
                flag=False
        
        image_info['refine_paras'][id]=refine_paras
        tqdm.write(f'{refine_paras}')

def ground2image():
    tiept_info['imgpt_x_bk']=np.zeros(np.size(tiept_info['imgpt_x']))
    tiept_info['imgpt_y_bk']=np.zeros(np.size(tiept_info['imgpt_y']))

    for id in tqdm(image_info['ImageID'],desc=f'backward intersection images'):
            indices=np.where(tiept_info['img_id']==id)[0]
            tiept_curr_img=dict(key=['img_id','imgpt_x','img_pt_y','object_name','objpt_x','objpt_y','objpt_z'])

            # tiept_curr_img=tiept_info.loc[indices]
            # tiept_curr_img=tiept_info[indices]
            for key in tiept_info.keys():
                tiept_curr_img[key]=tiept_info[key][indices]

            L=tiept_curr_img['objpt_x'].astype('float32')
            B=tiept_curr_img['objpt_y'].astype('float32')
            H=tiept_curr_img['objpt_z'].astype('float32')

            L=(L-image_info['longOffset'][id])/image_info['longScale'][id]
            B=(B-image_info['latOffset'][id])/image_info['latScale'][id]
            H=(H-image_info['heightOffset'][id])/image_info['heightScale'][id]

            BL=B*L;BH=B*H;LH=L*H;BB=B*B;LL=L*L;HH=H*H;BLH=B*L*H;BBL=B*B*L;BBH=B*B*H;LLB=L*L*B;LLH=L*L*H;HHB=H*H*B;HHL=H*H*L;BBB=B*B*B;LLL=L*L*L;HHH=H*H*H
            LNUM=image_info['lineNumCoef'][id][0]+image_info['lineNumCoef'][id][1]*L+image_info['lineNumCoef'][id][2]*B+image_info['lineNumCoef'][id][3]*H\
                        +image_info['lineNumCoef'][id][4]*BL+image_info['lineNumCoef'][id][5]*LH+image_info['lineNumCoef'][id][6]*BH\
                        +image_info['lineNumCoef'][id][7]*LL+image_info['lineNumCoef'][id][8]*BB+image_info['lineNumCoef'][id][9]*HH\
                        +image_info['lineNumCoef'][id][10]*BLH+image_info['lineNumCoef'][id][11]*LLL+image_info['lineNumCoef'][id][12]*BBL\
                        +image_info['lineNumCoef'][id][13]*HHL+image_info['lineNumCoef'][id][14]*LLB+image_info['lineNumCoef'][id][15]*BBB\
                        +image_info['lineNumCoef'][id][16]*HHB+image_info['lineNumCoef'][id][17]*LLH+image_info['lineNumCoef'][id][18]*BBH\
                        +image_info['lineNumCoef'][id][19]*HHH
                
            LDEN=image_info['lineDenCoef'][id][0]+image_info['lineDenCoef'][id][1]*L+image_info['lineDenCoef'][id][2]*B+image_info['lineDenCoef'][id][3]*H\
                    +image_info['lineDenCoef'][id][4]*BL+image_info['lineDenCoef'][id][5]*LH+image_info['lineDenCoef'][id][6]*BH\
                    +image_info['lineDenCoef'][id][7]*LL+image_info['lineDenCoef'][id][8]*BB+image_info['lineDenCoef'][id][9]*HH\
                    +image_info['lineDenCoef'][id][10]*BLH+image_info['lineDenCoef'][id][11]*LLL+image_info['lineDenCoef'][id][12]*BBL\
                    +image_info['lineDenCoef'][id][13]*HHL+image_info['lineDenCoef'][id][14]*LLB+image_info['lineDenCoef'][id][15]*BBB\
                    +image_info['lineDenCoef'][id][16]*HHB+image_info['lineDenCoef'][id][17]*LLH+image_info['lineDenCoef'][id][18]*BBH\
                    +image_info['lineDenCoef'][id][19]*HHH
                    
            SNUM=image_info['sampNumCoef'][id][0]+image_info['sampNumCoef'][id][1]*L+image_info['sampNumCoef'][id][2]*B+image_info['sampNumCoef'][id][3]*H\
                    +image_info['sampNumCoef'][id][4]*BL+image_info['sampNumCoef'][id][5]*LH+image_info['sampNumCoef'][id][6]*BH\
                    +image_info['sampNumCoef'][id][7]*LL+image_info['sampNumCoef'][id][8]*BB+image_info['sampNumCoef'][id][9]*HH\
                    +image_info['sampNumCoef'][id][10]*BLH+image_info['sampNumCoef'][id][11]*LLL+image_info['sampNumCoef'][id][12]*BBL\
                    +image_info['sampNumCoef'][id][13]*HHL+image_info['sampNumCoef'][id][14]*LLB+image_info['sampNumCoef'][id][15]*BBB\
                    +image_info['sampNumCoef'][id][16]*HHB+image_info['sampNumCoef'][id][17]*LLH+image_info['sampNumCoef'][id][18]*BBH\
                    +image_info['sampNumCoef'][id][19]*HHH
            
            SDEN=image_info['sampDenCoef'][id][0]+image_info['sampDenCoef'][id][1]*L+image_info['sampDenCoef'][id][2]*B+image_info['sampDenCoef'][id][3]*H\
                    +image_info['sampDenCoef'][id][4]*BL+image_info['sampDenCoef'][id][5]*LH+image_info['sampDenCoef'][id][6]*BH\
                    +image_info['sampDenCoef'][id][7]*LL+image_info['sampDenCoef'][id][8]*BB+image_info['sampDenCoef'][id][9]*HH\
                    +image_info['sampDenCoef'][id][10]*BLH+image_info['sampDenCoef'][id][11]*LLL+image_info['sampDenCoef'][id][12]*BBL\
                    +image_info['sampDenCoef'][id][13]*HHL+image_info['sampDenCoef'][id][14]*LLB+image_info['sampDenCoef'][id][15]*BBB\
                    +image_info['sampDenCoef'][id][16]*HHB+image_info['sampDenCoef'][id][17]*LLH+image_info['sampDenCoef'][id][18]*BBH\
                    +image_info['sampDenCoef'][id][19]*HHH
            
            l=(LNUM/LDEN)*image_info['lineScale'][id]+image_info['lineOffset'][id];s=(SNUM/SDEN)*image_info['sampScale'][id]+image_info['sampOffset'][id]
            tiept_info['imgpt_x_bk'][indices]=l;tiept_info['imgpt_y_bk'][indices]=s

def accuracy_assessment(refine_model='translation'):
    for id in tqdm(image_info['ImageID'],desc=f'accracy assessing images'):
            indices=np.where(tiept_info['img_id']==id)[0]
            tiept_curr_img=dict(key=['imgpt_id','object_name','img_id','imgpt_y','imgpt_x','objpt_x','objpt_y','objpt_z','imgpt_x_bk','imgpt_y_bk'])

            # tiept_curr_img=tiept_info.loc[indices]
            # tiept_curr_img=tiept_info[indices]
            for key in tiept_info.keys():
                tiept_curr_img[key]=tiept_info[key][indices]

            if refine_model=='translation':
                x_diff=tiept_curr_img['imgpt_x_bk']-tiept_curr_img['imgpt_x']-image_info['refine_paras'][id][0]
                y_diff=tiept_curr_img['imgpt_y_bk']-tiept_curr_img['imgpt_y']-image_info['refine_paras'][id][1]
            elif refine_model=='affine':
                x_diff=tiept_curr_img['imgpt_x_bk']-tiept_curr_img['imgpt_x']-image_info['refine_paras'][id][0]\
                    -image_info['refine_paras'][id][1]*tiept_curr_img['imgpt_x']-image_info['refine_paras'][id][2]*tiept_curr_img['imgpt_y']
                y_diff=tiept_curr_img['imgpt_y_bk']-tiept_curr_img['imgpt_y']-image_info['refine_paras'][id][3]\
                    -image_info['refine_paras'][id][4]*tiept_curr_img['imgpt_x']-image_info['refine_paras'][id][5]*tiept_curr_img['imgpt_y']

            x_diff_ave=np.mean(x_diff);y_diff_ave=np.mean(y_diff)
            tqdm.write(f'the average pixel error of image {id} in x direction: {x_diff_ave}')
            tqdm.write(f'the average pixel error of image {id} in y direction: {y_diff_ave}')

            x_diff_rmse=np.sqrt(np.mean((x_diff)**2));y_diff_rmse=np.sqrt(np.mean((y_diff)**2))
            tqdm.write(f'the RMSE of image {id} in x direction: {x_diff_rmse}')
            tqdm.write(f'the RMSE of image {id} in y direction: {y_diff_rmse}')

def format_writing_tiepts(output_file,format='xq'):
    if format=='xq':
        columns = ["Point ID", "X", "Y", "Z", "Image ID", "Imgpt_x", "Imgpt_y", "Reliability", "Type", "Overlap", "MaxBHR"]

        with open(output_file, "w") as file:
            tiept_num=len(tiept_info['img_id'])

            file.write("\tTie  point  Object  Coordinate\n".format())
            file.write("\tTie  point  Object  Num Is {}\n".format(tiept_num))
            file.write("\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(*columns))

            for i in tqdm(range(tiept_num),desc='writing tie points to file'):
                file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    i,
                    tiept_info["object_name"][i],
                    tiept_info["objpt_x"][i],
                    tiept_info["objpt_y"][i],
                    tiept_info["objpt_z"][i],
                    tiept_info["img_id"][i],
                    tiept_info["imgpt_x"][i],
                    tiept_info["imgpt_y"][i],
                    1,
                    1,
                    0,
                    0
                ))

def block_adjustment(refine_model='translation',adjust_method='with_laser'):
    ground_names=set(tiept_info['object_name'])

    if adjust_method=='optical':
        pass
    elif adjust_method=='with_laser':
        img_num=len(image_info['ImageID'])
        coeff_sum=np.zeros(1)
        const_sum=np.zeros(1)

        for ground_name in tqdm(ground_names,desc='adjusting by tiepoints'):
            idx=np.where(tiept_info['object_name']==ground_name)
            imgnum_of_tiept=len(tiept_info['img_id'][idx])

            L=tiept_info['objpt_x'][idx].astype('float32')
            B=tiept_info['objpt_y'][idx].astype('float32')
            H=tiept_info['objpt_z'][idx].astype('float32')

            img_id=tiept_info['img_id'][idx]

            L=(L-image_info['lon_off'][img_id])/image_info['lon_scale'][img_id]
            B=(B-image_info['lat_off'][img_id])/image_info['lat_scale'][img_id]
            H=(H-image_info['hei_off'][img_id])/image_info['hei_scale'][img_id]

            BL=B*L;BH=B*H;LH=L*H;BB=B*B;LL=L*L;HH=H*H;BLH=B*L*H;BBL=B*B*L;BBH=B*B*H;LLB=L*L*B;LLH=L*L*H;HHB=H*H*B;HHL=H*H*L;BBB=B*B*B;LLL=L*L*L;HHH=H*H*H
            LNUM=image_info['lineNumCoef'][img_id][:,0]+image_info['lineNumCoef'][img_id][:,1]*L+image_info['lineNumCoef'][img_id][:,2]*B+image_info['lineNumCoef'][img_id][:,3]*H\
                            +image_info['lineNumCoef'][img_id][:,4]*BL+image_info['lineNumCoef'][img_id][:,5]*LH+image_info['lineNumCoef'][img_id][:,6]*BH\
                            +image_info['lineNumCoef'][img_id][:,7]*LL+image_info['lineNumCoef'][img_id][:,8]*BB+image_info['lineNumCoef'][img_id][:,9]*HH\
                            +image_info['lineNumCoef'][img_id][:,10]*BLH+image_info['lineNumCoef'][img_id][:,11]*LLL+image_info['lineNumCoef'][img_id][:,12]*BBL\
                            +image_info['lineNumCoef'][img_id][:,13]*HHL+image_info['lineNumCoef'][img_id][:,14]*LLB+image_info['lineNumCoef'][img_id][:,15]*BBB\
                            +image_info['lineNumCoef'][img_id][:,16]*HHB+image_info['lineNumCoef'][img_id][:,17]*LLH+image_info['lineNumCoef'][img_id][:,18]*BBH\
                            +image_info['lineNumCoef'][img_id][:,19]*HHH
                    
            LDEN=image_info['lineDenCoef'][img_id][:,0]+image_info['lineDenCoef'][img_id][:,1]*L+image_info['lineDenCoef'][img_id][:,2]*B+image_info['lineDenCoef'][img_id][:,3]*H\
                    +image_info['lineDenCoef'][img_id][:,4]*BL+image_info['lineDenCoef'][img_id][:,5]*LH+image_info['lineDenCoef'][img_id][:,6]*BH\
                    +image_info['lineDenCoef'][img_id][:,7]*LL+image_info['lineDenCoef'][img_id][:,8]*BB+image_info['lineDenCoef'][img_id][:,9]*HH\
                    +image_info['lineDenCoef'][img_id][:,10]*BLH+image_info['lineDenCoef'][img_id][:,11]*LLL+image_info['lineDenCoef'][img_id][:,12]*BBL\
                    +image_info['lineDenCoef'][img_id][:,13]*HHL+image_info['lineDenCoef'][img_id][:,14]*LLB+image_info['lineDenCoef'][img_id][:,15]*BBB\
                    +image_info['lineDenCoef'][img_id][:,16]*HHB+image_info['lineDenCoef'][img_id][:,17]*LLH+image_info['lineDenCoef'][img_id][:,18]*BBH\
                    +image_info['lineDenCoef'][img_id][:,19]*HHH
                    
            SNUM=image_info['sampNumCoef'][img_id][:,0]+image_info['sampNumCoef'][img_id][:,1]*L+image_info['sampNumCoef'][img_id][:,2]*B+image_info['sampNumCoef'][img_id][:,3]*H\
                    +image_info['sampNumCoef'][img_id][:,4]*BL+image_info['sampNumCoef'][img_id][:,5]*LH+image_info['sampNumCoef'][img_id][:,6]*BH\
                    +image_info['sampNumCoef'][img_id][:,7]*LL+image_info['sampNumCoef'][img_id][:,8]*BB+image_info['sampNumCoef'][img_id][:,9]*HH\
                    +image_info['sampNumCoef'][img_id][:,10]*BLH+image_info['sampNumCoef'][img_id][:,11]*LLL+image_info['sampNumCoef'][img_id][:,12]*BBL\
                    +image_info['sampNumCoef'][img_id][:,13]*HHL+image_info['sampNumCoef'][img_id][:,14]*LLB+image_info['sampNumCoef'][img_id][:,15]*BBB\
                    +image_info['sampNumCoef'][img_id][:,16]*HHB+image_info['sampNumCoef'][img_id][:,17]*LLH+image_info['sampNumCoef'][img_id][:,18]*BBH\
                    +image_info['sampNumCoef'][img_id][:,19]*HHH
            
            SDEN=image_info['sampDenCoef'][img_id][:,0]+image_info['sampDenCoef'][img_id][:,1]*L+image_info['sampDenCoef'][img_id][:,2]*B+image_info['sampDenCoef'][img_id][:,3]*H\
                    +image_info['sampDenCoef'][img_id][:,4]*BL+image_info['sampDenCoef'][img_id][:,5]*LH+image_info['sampDenCoef'][img_id][:,6]*BH\
                    +image_info['sampDenCoef'][img_id][:,7]*LL+image_info['sampDenCoef'][img_id][:,8]*BB+image_info['sampDenCoef'][img_id][:,9]*HH\
                    +image_info['sampDenCoef'][img_id][:,10]*BLH+image_info['sampDenCoef'][img_id][:,11]*LLL+image_info['sampDenCoef'][img_id][:,12]*BBL\
                    +image_info['sampDenCoef'][img_id][:,13]*HHL+image_info['sampDenCoef'][img_id][:,14]*LLB+image_info['sampDenCoef'][img_id][:,15]*BBB\
                    +image_info['sampDenCoef'][img_id][:,16]*HHB+image_info['sampDenCoef'][img_id][:,17]*LLH+image_info['sampDenCoef'][img_id][:,18]*BBH\
                    +image_info['sampDenCoef'][img_id][:,19]*HHH
            
            dLNUMB=image_info['lineNumCoef'][img_id][:,2]+image_info['lineNumCoef'][img_id][:,4]*L+image_info['lineNumCoef'][img_id][:,6]*H+2*image_info['lineNumCoef'][img_id][:,8]*B+image_info['lineNumCoef'][img_id][:,10]*LH+2*image_info['lineNumCoef'][img_id][:,12]*BL+\
                image_info['lineNumCoef'][img_id][:,14]*LL+3*image_info['lineNumCoef'][img_id][:,15]*BB+image_info['lineNumCoef'][img_id][:,16]*HH+2*image_info['lineNumCoef'][img_id][:,18]*BH
            dLNUML=image_info['lineNumCoef'][img_id][:,1]+image_info['lineNumCoef'][img_id][:,4]*B+image_info['lineNumCoef'][img_id][:,5]*H+2*image_info['lineNumCoef'][img_id][:,7]*L+image_info['lineNumCoef'][img_id][:,10]*BH\
                +3*image_info['lineNumCoef'][img_id][:,11]*LL+image_info['lineNumCoef'][img_id][:,12]*BB+image_info['lineNumCoef'][img_id][:,13]*HH+2*image_info['lineNumCoef'][img_id][:,14]*BL+2*image_info['lineNumCoef'][img_id][:,17]*H*L
            dLNUMH=image_info['lineNumCoef'][img_id][:,3]+image_info['lineNumCoef'][img_id][:,5]*L+image_info['lineNumCoef'][img_id][:,6]*B+2*image_info['lineNumCoef'][img_id][:,9]*H\
                +image_info['lineNumCoef'][img_id][:,10]*BL+2*image_info['lineNumCoef'][img_id][:,13]*L*H+2*image_info['lineNumCoef'][img_id][:,16]*B*H+image_info['lineNumCoef'][img_id][:,17]*LL+image_info['lineNumCoef'][img_id][:,18]*BB+3*image_info['lineNumCoef'][img_id][:,19]*HH

            dLDENB=image_info['lineDenCoef'][img_id][:,2]+image_info['lineDenCoef'][img_id][:,4]*L+image_info['lineDenCoef'][img_id][:,6]*H+2*image_info['lineDenCoef'][img_id][:,8]*B+image_info['lineDenCoef'][img_id][:,10]*LH+2*image_info['lineDenCoef'][img_id][:,12]*BL+\
                image_info['lineDenCoef'][img_id][:,14]*LL+3*image_info['lineDenCoef'][img_id][:,15]*BB+image_info['lineDenCoef'][img_id][:,16]*HH+2*image_info['lineDenCoef'][img_id][:,18]*BH
            dLDENL=image_info['lineDenCoef'][img_id][:,1]+image_info['lineDenCoef'][img_id][:,4]*B+image_info['lineDenCoef'][img_id][:,5]*H+2*image_info['lineDenCoef'][img_id][:,7]*L+image_info['lineDenCoef'][img_id][:,10]*BH\
                +3*image_info['lineDenCoef'][img_id][:,11]*LL+image_info['lineDenCoef'][img_id][:,12]*BB+image_info['lineDenCoef'][img_id][:,13]*HH+2*image_info['lineDenCoef'][img_id][:,14]*BL+2*image_info['lineDenCoef'][img_id][:,17]*H*L
            dLDENH=image_info['lineDenCoef'][img_id][:,3]+image_info['lineDenCoef'][img_id][:,5]*L+image_info['lineDenCoef'][img_id][:,6]*B+2*image_info['lineDenCoef'][img_id][:,9]*H\
                +image_info['lineDenCoef'][img_id][:,10]*BL+2*image_info['lineDenCoef'][img_id][:,13]*L*H+2*image_info['lineDenCoef'][img_id][:,16]*B*H+image_info['lineDenCoef'][img_id][:,17]*LL+image_info['lineDenCoef'][img_id][:,18]*BB+3*image_info['lineDenCoef'][img_id][:,19]*HH

            dSNUMB=image_info['sampNumCoef'][img_id][:,2]+image_info['sampNumCoef'][img_id][:,4]*L+image_info['sampNumCoef'][img_id][:,6]*H+2*image_info['sampNumCoef'][img_id][:,8]*B+image_info['sampNumCoef'][img_id][:,10]*LH+2*image_info['sampNumCoef'][img_id][:,12]*BL+\
                image_info['sampNumCoef'][img_id][:,14]*LL+3*image_info['sampNumCoef'][img_id][:,15]*BB+image_info['sampNumCoef'][img_id][:,16]*HH+2*image_info['sampNumCoef'][img_id][:,18]*BH
            dSNUML=image_info['sampNumCoef'][img_id][:,1]+image_info['sampNumCoef'][img_id][:,4]*B+image_info['sampNumCoef'][img_id][:,5]*H+2*image_info['sampNumCoef'][img_id][:,7]*L+image_info['sampNumCoef'][img_id][:,10]*BH\
                +3*image_info['sampNumCoef'][img_id][:,11]*LL+image_info['sampNumCoef'][img_id][:,12]*BB+image_info['sampNumCoef'][img_id][:,13]*HH+2*image_info['sampNumCoef'][img_id][:,14]*BL+2*image_info['sampNumCoef'][img_id][:,17]*H*L
            dSNUMH=image_info['sampNumCoef'][img_id][:,3]+image_info['sampNumCoef'][img_id][:,5]*L+image_info['sampNumCoef'][img_id][:,6]*B+2*image_info['sampNumCoef'][img_id][:,9]*H\
                +image_info['sampNumCoef'][img_id][:,10]*BL+2*image_info['sampNumCoef'][img_id][:,13]*L*H+2*image_info['sampNumCoef'][img_id][:,16]*B*H+image_info['sampNumCoef'][img_id][:,17]*LL+image_info['sampNumCoef'][img_id][:,18]*BB+3*image_info['sampNumCoef'][img_id][:,19]*HH

            dSDENB=image_info['sampDenCoef'][img_id][:,2]+image_info['sampDenCoef'][img_id][:,4]*L+image_info['sampDenCoef'][img_id][:,6]*H+2*image_info['sampDenCoef'][img_id][:,8]*B+image_info['sampDenCoef'][img_id][:,10]*LH+2*image_info['sampDenCoef'][img_id][:,12]*BL+\
                image_info['sampDenCoef'][img_id][:,14]*LL+3*image_info['sampDenCoef'][img_id][:,15]*BB+image_info['sampDenCoef'][img_id][:,16]*HH+2*image_info['sampDenCoef'][img_id][:,18]*BH
            dSDENL=image_info['sampDenCoef'][img_id][:,1]+image_info['sampDenCoef'][img_id][:,4]*B+image_info['sampDenCoef'][img_id][:,5]*H+2*image_info['sampDenCoef'][img_id][:,7]*L+image_info['sampDenCoef'][img_id][:,10]*BH\
                +3*image_info['sampDenCoef'][img_id][:,11]*LL+image_info['sampDenCoef'][img_id][:,12]*BB+image_info['sampDenCoef'][img_id][:,13]*HH+2*image_info['sampDenCoef'][img_id][:,14]*BL+2*image_info['sampDenCoef'][img_id][:,17]*H*L
            dSDENH=image_info['sampDenCoef'][img_id][:,3]+image_info['sampDenCoef'][img_id][:,5]*L+image_info['sampDenCoef'][img_id][:,6]*B+2*image_info['sampDenCoef'][img_id][:,9]*H\
                +image_info['sampDenCoef'][img_id][:,10]*BL+2*image_info['sampDenCoef'][img_id][:,13]*L*H+2*image_info['sampDenCoef'][img_id][:,16]*B*H+image_info['sampDenCoef'][img_id][:,17]*LL+image_info['sampDenCoef'][img_id][:,18]*BB+3*image_info['sampDenCoef'][img_id][:,19]*HH

            dLB=(dLNUMB*LDEN-LNUM*dLDENB)/(LDEN*LDEN)
            dLL=(dLNUML*LDEN-LNUM*dLDENL)/(LDEN*LDEN)
            dLH=(dLNUMH*LDEN-LNUM*dLDENH)/(LDEN*LDEN)

            dSB=(dSNUMB*SDEN-SNUM*dSDENB)/(SDEN*SDEN)
            dSL=(dSNUML*SDEN-SNUM*dSDENL)/(SDEN*SDEN)
            dSH=(dSNUMH*SDEN-SNUM*dSDENH)/(SDEN*SDEN)

            dLlat=dLB/image_info['latScale'][img_id];dLlon=dLL/image_info['longScale'][img_id];dLhei=dLH/image_info['heightScale'][img_id]
            dSlat=dSB/image_info['latScale'][img_id];dSlon=dSL/image_info['longScale'][img_id];dShei=dSH/image_info['heightScale'][img_id]

            l=np.asarray(tiept_info['imgpt_x'][idx])
            s=np.asarray(tiept_info['imgpt_y'][idx])

            if refine_model=='translation':
                refine_num=2
            elif refine_model=='affine':
                refine_num=6

            A1=np.zeros((imgnum_of_tiept*2,img_num*refine_num))
            A2=np.zeros((imgnum_of_tiept*2,3))
            P=np.eye(imgnum_of_tiept*2)
            f=np.zeros((imgnum_of_tiept*2))      
            refine_paras=np.zeros((img_num*refine_num))
            X=np.zeros((img_num*refine_num))

            single_num=int(refine_num/2)
            if refine_model=='translation':
                for i in range(imgnum_of_tiept):
                    id=tiept_info['img_id'][i]
                    A1[2*i,id*single_num:id*single_num+single_num]=-1
                    A1[2*i+1,id*single_num:id*single_num+single_num]=-1
                f[0:imgnum_of_tiept*2:2]=tiept_info['imgpt_x'][idx]+refine_paras[0]-((LNUM/LDEN)*image_info['lineScale'][img_id]+image_info['lineOffset'][img_id])
                f[1:imgnum_of_tiept*2+1:2]=tiept_info['imgpt_y'][idx]+refine_paras[1]-((SNUM/SDEN)*image_info['sampScale'][img_id]+image_info['sampOffset'][img_id])
            elif refine_model=='affine':
                for i in range(imgnum_of_tiept):
                    id=tiept_info['img_id'][i]
                    A1[2*i,id*single_num:id*single_num+single_num]=-1
                    A1[2*i+1,id*single_num:id*single_num+single_num]=-1
                f[0:imgnum_of_tiept*2:2]=tiept_info['imgpt_x'][idx]+refine_paras[0]+refine_paras[1]*tiept_info['imgpt_x'][idx]+refine_paras[2]*tiept_info['imgpt_y'][idx]\
                    -((LNUM/LDEN)*image_info['lineScale'][img_id]+image_info['lineOffset'][img_id])
                f[1:imgnum_of_tiept*2+1:2]=tiept_info['imgpt_y'][idx]+refine_paras[3]+refine_paras[4]*tiept_info['imgpt_x'][idx]+refine_paras[5]*tiept_info['imgpt_y'][idx]\
                    -((SNUM/SDEN)*image_info['sampScale'][img_id]+image_info['sampOffset'][img_id])
            A2[0:-1:2,:]=np.asarray([dLlon*image_info['lineScale'][img_id],dLlat*image_info['lineScale'][img_id],dLhei*image_info['line_scale'][img_id]])
            A2[1:-1+1:2,:]=np.asarray([dSlon*image_info['sampScale'][img_id],dSlat*image_info['sampScale'][img_id],dShei*image_info['sampScale'][img_id]])

            sns.heatmap(A1);sns.heatmap(A2);sns.heatmap(f)

            coeff=np.linalg.inv(A1.T@P@A1-A1.T@P@A2@np.linalg.inv(A2.T@P@A2)@A2.T@P@A1)
            const=A1@P@f-A1.T@P@A2@np.linalg.inv(A2.T@P@A2)@A2.T@P@f
            Q_xx=np.linalg.inv(coeff)

            iter_count+=1

if __name__=='__main__':
    s=time.perf_counter()
    ############ pre-defined parameters ############
    paras=np.asarray([4,3,2,1,5]).reshape((5,1))
    paras_num=len(paras)
    max_core=cpu_count()
    counts=0
    ################################################

    ############ files ############
    order_file=r"F:\\ToZY\\zdb.tri.txt"
    tiept_image_file=r"F:\\ToZY\\tie.txt"
    tiept_out_file=r'F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\tiept_xq_parallel.txt'
    sla_file=r'F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\tie.sla'
    ################################################

    order_info,image_info=load_order_file(order_file)
    load_rpc_file(type='txt')
    tiept_num,tiept_info=load_imgtiept_data(tiept_image_file)
    # format_writing_tiepts(tiept_info,tiept_out_file)
    # slapt_num,slapt_info=load_sla_file(sla_file)

    s=time.perf_counter()
    forward_intersec_on_const_level()
    e=time.perf_counter()
    print(e-s)

    # tiept_info=tiept_info.loc[tiept_info['objpt_x']!=0]
    # tiept_info=tiept_info[tiept_info['objpt_x']!=0]
    # tiept_info.index=range(len(tiept_info))
    # indices=tiept_info['objpt_x']!=0
    # for key in tiept_info.keys():
    #     tiept_info[key]=tiept_info[key][indices]

    # tiept_info=pd.read_csv(tiept_out_file,sep='\t')
    # tiept_info.columns=['object_name','objpt_x','objpt_y','objpt_z','img_id','imgpt_x','imgpt_y','Reliability','Type','Overlap','MaxBHR']
    # del tiept_info['Reliability'],tiept_info['Type'],tiept_info['Overlap'],tiept_info['MaxBHR']
    
    # tiept_info = {col:tiept_info[col].tolist() for col in tiept_info.columns}
    # for key in tiept_info.keys():
    #     tiept_info[key]=np.asarray(tiept_info[key])

    forward_intersec()

    # refine_para_compute(refine_model='translation')
    # ground2image()
    # accuracy_assessment(refine_model='translation')
    # # block_adjustment(tiept_info,tiept_num,image_info,refine_model='translation',adjust_method='with_laser')
    # e=time.perf_counter()
    # print((e-s)/60)
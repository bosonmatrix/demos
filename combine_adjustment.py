#%% preparations
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

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
        
    image_info['rows']=[]
    image_info['cols']=[]

    for image_file in image_info['ImageFile']:
        with open(image_file) as f:
            lines=f.readlines()
            row=str.split(lines[0].strip(),':')[1]
            col=str.split(lines[1].strip(),':')[1]
            image_info['rows'].append(eval(row))
            image_info['cols'].append(eval(col))

    for key in image_info.keys():
        image_info[key]=np.asarray(image_info[key])

    return order_info,image_info

def load_groundtiept_data(tiept_file):
    with open(tiept_file, 'r') as file:
        lines = file.readlines()

    keys=['imgpt_id','object_name','img_id','objpt_x','objpt_y','objpt_z']
    tiept_info = {}

    for line in tqdm(lines,desc='loading tie points'):
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

    tiept_num=len(tiept_info['object_name'])
    tiept_info['imgpt_x']=np.zeros(tiept_num)
    tiept_info['imgpt_y']=np.zeros(tiept_num)

    return tiept_num,tiept_info

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

    tiept_num=len(tiept_info['object_name'])
    tiept_info['objpt_x']=np.zeros(tiept_num)
    tiept_info['objpt_y']=np.zeros(tiept_num)
    tiept_info['objpt_z']=np.zeros(tiept_num)

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

    keys=['imgpt_id','object_name','img_id','objpt_x','objpt_y','objpt_z','imgpt_y','imgpt_x','reliability','type']
    slapt_info = {}
    slapt_num=eval(lines[0])

    for line in tqdm(lines[1:-1]):
        line = line.strip()

        if line:
            values = line.split('\t')
            for i in range(len(keys)):
                if keys[i] not in slapt_info.keys():
                    if keys[i]!='object_name':
                        slapt_info[keys[i]]=[eval(values[i])]
                    else:
                        slapt_info[keys[i]]=[values[i]]
                else:
                    if keys[i]!='object_name':
                        slapt_info[keys[i]].append(eval(values[i]))
                    else:
                        slapt_info[keys[i]].append(values[i])

    for key in slapt_info.keys():
        slapt_info[key]=np.asarray(slapt_info[key])

    return slapt_num,slapt_info

def load_accuracy_file(accu_file):
    with open(accu_file) as f:
        lines=f.readlines()
        line_idx=0

        gcpt_info['vx']=np.asarray([-9999.9]*len(gcpt_info['imgpt_x']))
        gcpt_info['vy']=np.asarray([-9999.9]*len(gcpt_info['imgpt_x']))

        for i in tqdm(range(4,len(lines)),desc='reading check gcp points\' accuracy'):
            if lines[i]=='\n':
                line_idx=i
                break
            elif lines[i].find('PointID')!=-1:
                continue
            elif lines[i].find('RMSE')!=-1 or lines[i].find('MEAN')!=-1 or lines[i].find('ERR')!=-1 or lines[i].find('MIN')!=-1 or lines[i].find('MAX')!=-1:
                line_split=str.split(lines[i].strip(),sep=':')
                if lines[i].find('nan')!=-1:
                    gcpt_info[line_split[0]]=np.nan
                else:  
                    gcpt_info[line_split[0]]=[eval(line_split[1])]
            else:
                line_split=str.split(lines[i].strip(),sep='\t')
                idx1=np.where(gcpt_info['object_name']==line_split[0])
                idx2=np.where(gcpt_info['img_id']==eval(line_split[1]))
                idx=np.intersect1d(idx1,idx2)
                if line_split[2]=='-nan(ind)' and line_split[3]!='-nan(ind)':
                    gcpt_info['vx'][idx]=np.nan
                    gcpt_info['vy'][idx]=eval(line_split[3])
                elif line_split[2]!='-nan(ind)' and line_split[3]=='-nan(ind)':
                    gcpt_info['vx'][idx]=eval(line_split[2])
                    gcpt_info['vy'][idx]=np.nan
                elif line_split[2]=='-nan(ind)' and line_split[3]=='-nan(ind)':
                    gcpt_info['vx'][idx]=np.nan
                    gcpt_info['vy'][idx]=np.nan
                else:
                    gcpt_info['vx'][idx]=eval(line_split[2])
                    gcpt_info['vy'][idx]=eval(line_split[3])
        
        for i in tqdm(range(line_idx+4,len(lines)),desc='reading control gcp points\' accuracy'):
            if lines[i]=='\n':
                line_idx=i
                break
            elif lines[i].find('PointID')!=-1:
                continue
            elif lines[i].find('RMSE')!=-1 or lines[i].find('MEAN')!=-1 or lines[i].find('ERR')!=-1 or lines[i].find('MIN')!=-1 or lines[i].find('MAX')!=-1:
                line_split=str.split(lines[i].strip(),sep=':')
                if lines[i].find('nan')!=-1:
                    gcpt_info[line_split[0]]=np.nan
                else: 
                    gcpt_info[line_split[0]]=[eval(line_split[1])]
            else:
                line_split=str.split(lines[i].strip(),sep='\t')
                idx1=np.where(gcpt_info['object_name']==line_split[0])
                idx2=np.where(gcpt_info['img_id']==eval(line_split[1]))
                idx=np.intersect1d(idx1,idx2)
                if line_split[2]=='-nan(ind)' and line_split[3]!='-nan(ind)':
                    gcpt_info['vx'][idx]=np.nan
                    gcpt_info['vy'][idx]=eval(line_split[3])
                elif line_split[2]!='-nan(ind)' and line_split[3]=='-nan(ind)':
                    gcpt_info['vx'][idx]=eval(line_split[2])
                    gcpt_info['vy'][idx]=np.nan
                elif line_split[2]=='-nan(ind)' and line_split[3]=='-nan(ind)':
                    gcpt_info['vx'][idx]=np.nan
                    gcpt_info['vy'][idx]=np.nan
                else:
                    gcpt_info['vx'][idx]=eval(line_split[2])
                    gcpt_info['vy'][idx]=eval(line_split[3])

        tiept_info['vx']=np.asarray([-9999.9]*len(tiept_info['imgpt_x']))
        tiept_info['vy']=np.asarray([-9999.9]*len(tiept_info['imgpt_x']))

        for i in tqdm(range(line_idx+4,len(lines)),desc='reading tie points\' accuracy'):
            if lines[i]=='\n':
                line_idx=i
                break
            elif lines[i].find('PointID')!=-1:
                continue
            elif lines[i].find('RMSE')!=-1 or lines[i].find('MEAN')!=-1 or lines[i].find('ERR')!=-1 or lines[i].find('MIN')!=-1 or lines[i].find('MAX')!=-1:
                line_split=str.split(lines[i].strip(),sep=':')
                if lines[i].find('nan')!=-1:
                    tiept_info[line_split[0]]=np.nan
                else:
                    tiept_info[line_split[0]]=[eval(line_split[1])]
            else:
                line_split=str.split(lines[i].strip(),sep='\t')
                idx1=np.where(tiept_info['object_name']==line_split[0])[0]
                idx2=np.where(tiept_info['img_id']==eval(line_split[1]))[0]
                idx=np.intersect1d(idx1,idx2)
                if line_split[2]=='-nan(ind)' and line_split[3]!='-nan(ind)':
                    tiept_info['vx'][idx]=np.nan
                    tiept_info['vy'][idx]=eval(line_split[3])
                elif line_split[2]!='-nan(ind)' and line_split[3]=='-nan(ind)':
                    tiept_info['vx'][idx]=eval(line_split[2])
                    tiept_info['vy'][idx]=np.nan
                elif line_split[2]=='-nan(ind)' and line_split[3]=='-nan(ind)':
                    tiept_info['vx'][idx]=np.nan
                    tiept_info['vy'][idx]=np.nan
                else:
                    tiept_info['vx'][idx]=eval(line_split[2])
                    tiept_info['vy'][idx]=eval(line_split[3])

        slapt_info['vx']=np.asarray([-9999.9]*len(slapt_info['imgpt_x']))
        slapt_info['vy']=np.asarray([-9999.9]*len(slapt_info['imgpt_x']))

        for i in tqdm(range(line_idx+4,len(lines)),desc='reading check sla points\' accuracy'):
            if lines[i]=='\n':
                line_idx=i
                break
            elif lines[i].find('PointID')!=-1:
                continue
            elif lines[i].find('RMSE')!=-1 or lines[i].find('MEAN')!=-1 or lines[i].find('ERR')!=-1 or lines[i].find('MIN')!=-1 or lines[i].find('MAX')!=-1:
                line_split=str.split(lines[i].strip(),sep=':')
                if lines[i].find('nan')!=-1:
                    slapt_info[line_split[0]]=np.nan
                else:  
                    slapt_info[line_split[0]]=[eval(line_split[1])]
            else:
                line_split=str.split(lines[i].strip(),sep='\t')
                idx1=np.where(slapt_info['object_name']==line_split[0])
                idx2=np.where(slapt_info['img_id']==eval(line_split[1]))
                idx=np.intersect1d(idx1,idx2)
                if line_split[2]=='-nan(ind)' and line_split[3]!='-nan(ind)':
                    slapt_info['vx'][idx]=np.nan
                    slapt_info['vy'][idx]=eval(line_split[3])
                elif line_split[2]!='-nan(ind)' and line_split[3]=='-nan(ind)':
                    slapt_info['vx'][idx]=eval(line_split[2])
                    slapt_info['vy'][idx]=np.nan
                elif line_split[2]=='-nan(ind)' and line_split[3]=='-nan(ind)':
                    slapt_info['vx'][idx]=np.nan
                    slapt_info['vy'][idx]=np.nan
                else:
                    slapt_info['vx'][idx]=eval(line_split[2])
                    slapt_info['vy'][idx]=eval(line_split[3])

        for i in tqdm(range(line_idx+4,len(lines)),desc='reading control sla points\' accuracy'):
            if lines[i]=='\n':
                line_idx=i
                break
            elif lines[i].find('PointID')!=-1:
                continue
            elif lines[i].find('RMSE')!=-1 or lines[i].find('MEAN')!=-1 or lines[i].find('ERR')!=-1 or lines[i].find('MIN')!=-1 or lines[i].find('MAX')!=-1:
                line_split=str.split(lines[i].strip(),sep=':')
                if lines[i].find('nan')!=-1:
                    slapt_info[line_split[0]]=np.nan
                else:  
                    slapt_info[line_split[0]]=[eval(line_split[1])]
            else:
                line_split=str.split(lines[i].strip(),sep='\t')
                idx1=np.where(slapt_info['object_name']==line_split[0])
                idx2=np.where(slapt_info['img_id']==eval(line_split[1]))
                idx=np.intersect1d(idx1,idx2)
                if line_split[2]=='-nan(ind)' and line_split[3]!='-nan(ind)':
                    slapt_info['vx'][idx]=np.nan
                    slapt_info['vy'][idx]=eval(line_split[3])
                elif line_split[2]!='-nan(ind)' and line_split[3]=='-nan(ind)':
                    slapt_info['vx'][idx]=eval(line_split[2])
                    slapt_info['vy'][idx]=np.nan
                elif line_split[2]=='-nan(ind)' and line_split[3]=='-nan(ind)':
                    slapt_info['vx'][idx]=np.nan
                    slapt_info['vy'][idx]=np.nan
                else:
                    slapt_info['vx'][idx]=eval(line_split[2])
                    slapt_info['vy'][idx]=eval(line_split[3])

        dompt_info['vx']=np.asarray([-9999.9]*len(dompt_info['imgpt_x']))
        dompt_info['vy']=np.asarray([-9999.9]*len(dompt_info['imgpt_x']))

        for i in tqdm(range(line_idx+4,len(lines)),desc='reading check dom points\' accuracy'):
            if lines[i]=='\n':
                line_idx=i
                break
            elif lines[i].find('PointID')!=-1:
                continue
            elif lines[i].find('RMSE')!=-1 or lines[i].find('MEAN')!=-1 or lines[i].find('ERR')!=-1 or lines[i].find('MIN')!=-1 or lines[i].find('MAX')!=-1:
                line_split=str.split(lines[i].strip(),sep=':')
                if lines[i].find('nan')!=-1:
                    dompt_info[line_split[0]]=np.nan
                else:  
                    dompt_info[line_split[0]]=[eval(line_split[1])]
            else:
                line_split=str.split(lines[i].strip(),sep='\t')
                idx1=np.where(dompt_info['object_name']==line_split[0])
                idx2=np.where(dompt_info['img_id']==eval(line_split[1]))
                idx=np.intersect1d(idx1,idx2)
                if line_split[2]=='-nan(ind)' and line_split[3]!='-nan(ind)':
                    dompt_info['vx'][idx]=np.nan
                    dompt_info['vy'][idx]=eval(line_split[3])
                elif line_split[2]!='-nan(ind)' and line_split[3]=='-nan(ind)':
                    dompt_info['vx'][idx]=eval(line_split[2])
                    dompt_info['vy'][idx]=np.nan
                elif line_split[2]=='-nan(ind)' and line_split[3]=='-nan(ind)':
                    dompt_info['vx'][idx]=np.nan
                    dompt_info['vy'][idx]=np.nan
                else:
                    dompt_info['vx'][idx]=eval(line_split[2])
                    dompt_info['vy'][idx]=eval(line_split[3])
        
        for i in tqdm(range(line_idx+4,len(lines)),desc='reading control dom points\' accuracy'):
            if lines[i]=='\n':
                line_idx=i
                break
            elif lines[i].find('PointID')!=-1:
                continue
            elif lines[i].find('RMSE')!=-1 or lines[i].find('MEAN')!=-1 or lines[i].find('ERR')!=-1 or lines[i].find('MIN')!=-1 or lines[i].find('MAX')!=-1:
                line_split=str.split(lines[i].strip(),sep=':')
                if lines[i].find('nan')!=-1:
                    dompt_info[line_split[0]]=np.nan
                else:  
                    dompt_info[line_split[0]]=[eval(line_split[1])]
            else:
                line_split=str.split(lines[i].strip(),sep='\t')
                idx1=np.where(dompt_info['object_name']==line_split[0])
                idx2=np.where(dompt_info['img_id']==eval(line_split[1]))
                idx=np.intersect1d(idx1,idx2)
                if line_split[2]=='-nan(ind)' and line_split[3]!='-nan(ind)':
                    dompt_info['vx'][idx]=np.nan
                    dompt_info['vy'][idx]=eval(line_split[3])
                elif line_split[2]!='-nan(ind)' and line_split[3]=='-nan(ind)':
                    dompt_info['vx'][idx]=eval(line_split[2])
                    dompt_info['vy'][idx]=np.nan
                elif line_split[2]=='-nan(ind)' and line_split[3]=='-nan(ind)':
                    dompt_info['vx'][idx]=np.nan
                    dompt_info['vy'][idx]=np.nan
                else:
                    dompt_info['vx'][idx]=eval(line_split[2])
                    dompt_info['vy'][idx]=eval(line_split[3])

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

def accuracy_visulization(pt_info,save_path=None):
    sns.set(style='whitegrid')
    figsize_w=16;figsize_h=16
    font_size=20
    intervals=100
    min_tick_num=100

    pt_info = {key: value for key, value in pt_info.items() if key not in ['RMSE_X','RMSE_Y','MEAN_X','MEAN_Y','ERR_X','ERR_Y','MIN_X','MIN_Y','MAX_X','MAX_Y']}
    ptinfo_df=pd.DataFrame(pt_info)
    ptinfo_df=ptinfo_df.loc[ptinfo_df['vx']!=-9999.9]

    plt.figure(figsize=(figsize_w, figsize_h))
    sns.scatterplot(x='imgpt_x',y='vx',hue='img_id',size=0.01,data=ptinfo_df,style='img_id',legend='auto')
    plt.title('x residuals',fontsize=font_size)
    plt.xlabel('x',fontsize=font_size);plt.ylabel('y',fontsize=font_size)
    plt.savefig(os.path.join(save_path,'x_residuals.png'),bbox_inches='tight')

    plt.figure(figsize=(figsize_w, figsize_h))
    sns.scatterplot(x='imgpt_y',y='vy',hue='img_id',size=0.01,data=ptinfo_df,style='img_id',legend='auto')
    plt.title('y residuals',fontsize=font_size)
    plt.xlabel('x',fontsize=font_size);plt.ylabel('y',fontsize=font_size)
    plt.savefig(os.path.join(save_path,'y_residuals.png'),bbox_inches='tight')

    for img in set(image_info['ImageID']):
        indices=ptinfo_df['img_id']==img
        x_plt=ptinfo_df['imgpt_x'][indices]
        y_plt=ptinfo_df['imgpt_y'][indices]
        u_plt=ptinfo_df['vx'][indices]
        v_plt=ptinfo_df['vy'][indices]
        ptinfo_df["object_name_num"] = pd.factorize(ptinfo_df["object_name"])[0].astype(int)
        colors=ptinfo_df['object_name_num'][indices]
        plt.figure(figsize=(figsize_w, figsize_h))
        axes=plt.axes()
        plt.quiver(x_plt[0:len(x_plt):intervals],y_plt[0:len(x_plt):intervals],u_plt[0:len(x_plt):intervals],v_plt[0:len(x_plt):intervals],colors[0:len(x_plt):intervals],cmap='turbo',width=1e-3)
        plt.title(f'horizontal shift (image {img})',fontsize=font_size)
        plt.xlim((0,image_info['cols'][img]));plt.ylim((0,image_info['rows'][img]))
        plt.xlabel('sample',fontsize=font_size);plt.ylabel('line',fontsize=font_size)
        minor_ticks_x=np.linspace(0,image_info['cols'][img],min_tick_num)
        minor_ticks_y=np.linspace(0,image_info['rows'][img],min_tick_num)
        axes.set_xticks(minor_ticks_x, minor=True)
        axes.set_yticks(minor_ticks_y, minor=True)
        axes.grid(which="minor", alpha=0.3)
        plt.savefig(os.path.join(save_path,f'horizontal_shift_image_{img}.png'),bbox_inches='tight')

    for img in set(image_info['ImageID']):
        indices=ptinfo_df['img_id']==img
        x_plt=ptinfo_df['imgpt_x'][indices]
        y_plt=ptinfo_df['imgpt_y'][indices]
        u_plt=ptinfo_df['vx'][indices]
        v_plt=ptinfo_df['vy'][indices]
        ptinfo_df["object_name_num"] = pd.factorize(ptinfo_df["object_name"])[0].astype(int)
        colors=ptinfo_df['object_name_num'][indices]
        plt.figure(figsize=(figsize_w, figsize_h))
        axes=plt.axes()
        plt.quiver(x_plt[0:len(x_plt):intervals],y_plt[0:len(x_plt):intervals],u_plt[0:len(x_plt):intervals],0,colors[0:len(x_plt):intervals],cmap='turbo',width=1e-3)
        plt.title(f'sample shift (image {img})',fontsize=font_size)
        plt.xlim((0,image_info['cols'][img]));plt.ylim((0,image_info['rows'][img]))
        plt.xlabel('sample',fontsize=font_size);plt.ylabel('line',fontsize=font_size)
        minor_ticks_x=np.linspace(0,image_info['cols'][img],min_tick_num)
        minor_ticks_y=np.linspace(0,image_info['rows'][img],min_tick_num)
        axes.set_xticks(minor_ticks_x, minor=True)
        axes.set_yticks(minor_ticks_y, minor=True)
        axes.grid(which="minor", alpha=0.3)
        plt.savefig(os.path.join(save_path,f'column_shift_image_{img}.png'),bbox_inches='tight')
    
    for img in set(image_info['ImageID']):
        indices=ptinfo_df['img_id']==img
        x_plt=ptinfo_df['imgpt_x'][indices]
        y_plt=ptinfo_df['imgpt_y'][indices]
        u_plt=ptinfo_df['vx'][indices]
        v_plt=ptinfo_df['vy'][indices]
        ptinfo_df["object_name_num"] = pd.factorize(ptinfo_df["object_name"])[0].astype(int)
        colors=ptinfo_df['object_name_num'][indices]
        plt.figure(figsize=(figsize_w, figsize_w))
        axes=plt.axes()
        plt.quiver(x_plt[1:len(x_plt):intervals],y_plt[1:len(x_plt):intervals],0,v_plt[1:len(x_plt):intervals],colors[0:len(x_plt):intervals],cmap='turbo',width=1e-3)
        plt.title(f'line shift (image {img})',fontsize=font_size)
        plt.xlim((0,image_info['cols'][img]));plt.ylim((0,image_info['rows'][img]))
        plt.xlabel('sample',fontsize=font_size);plt.ylabel('line',fontsize=font_size)
        minor_ticks_x=np.linspace(0,image_info['cols'][img],min_tick_num)
        minor_ticks_y=np.linspace(0,image_info['rows'][img],min_tick_num)
        axes.set_xticks(minor_ticks_x, minor=True)
        axes.set_yticks(minor_ticks_y, minor=True)
        axes.grid(which="minor", alpha=0.3)
        plt.savefig(os.path.join(save_path,f'row_shift_image_{img}.png'),bbox_inches='tight')

#%% pre-defined parameters
if __name__=='__main__':
    gcpt_info={'imgpt_x':[0],'imgpt_y':[0]};dompt_info={'imgpt_x':[0],'imgpt_y':[0]}

    order_file=r"F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\zdb.tri.txt"
    tiept_image_file=r"F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\zdb.tiepick-ties.tie"
    tiept_ground_file=r"F:\\ToZY\\all-L1.tie.txt"
    tiept_out_file=r'F:\\ToZY\\tiept_xq.txt'
    sla_file=r"F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\tie.sla"
    accu_file=r"F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\result_freenet\\Residual\\Post\\ImgResidual.txt"
    figure_save_path=r'F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\result_freenet\\figures'

#%% load files
    order_info,image_info=load_order_file(order_file)
    # load_rpc_file(type='txt')
    # print(image_info)
    tiept_num,tiept_info=load_imgtiept_data(tiept_image_file)
    slapt_num,slapt_info=load_sla_file(sla_file)
    # ground2image()

    # tiept_info['imgpt_x']=tiept_info['imgpt_x_bk']
    # tiept_info['imgpt_y']=tiept_info['imgpt_y_bk']

    # indices=[]
    # for img_id in image_info['ImageID']:    
    #     idx1=tiept_info['img_id']==img_id
    #     idx2=(0<=tiept_info['imgpt_x'])&(tiept_info['imgpt_x']<=(image_info['cols'][img_id]-1))
    #     idx3=(0<=tiept_info['imgpt_y'])&(tiept_info['imgpt_y']<=(image_info['rows'][img_id]-1))
    #     indices.append((idx2&idx3))
    # indices=indices[0]|indices[1]
    # for key in tiept_info.keys():
    #     tiept_info[key]=tiept_info[key][indices]

    # format_writing_tiepts(tiept_out_file)
#%% accuracy assessment  
    load_accuracy_file(accu_file)

    accuracy_visulization(tiept_info,figure_save_path)
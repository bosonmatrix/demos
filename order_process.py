#%% preparations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from concurrent import futures
import warnings
import time
import pyproj
from sklearn import neighbors
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

    for line in lines[11:len(lines)]:
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

def load_errtiept_data(err_tiept_file):
    with open(err_tiept_file,'r') as f:
        lines=f.readlines()
        err_name=[each.strip() for each in lines[1:len(lines)]]
    
    return err_name

def load_imgtiept_data(err_name,imgtiept_file):
    with open(imgtiept_file, 'r') as file:
        lines = file.readlines()

    keys=['imgpt_id','object_name','img_id','imgpt_y','imgpt_x']
    tiept_info = {}

    for line in tqdm(lines[1:len(lines)],desc='loading image-tie points'):
        line = line.strip()

        if line:
            values = line.split('\t')
            if values[1] not in err_name:
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
    tiept_info['objpt_z']=np.full(tiept_num,-9999)
    tiept_info['overlap']=np.zeros(tiept_num)

    return tiept_num,tiept_info

def load_objtiept_data(err_name,objtiept_file):
    with open(objtiept_file,'r') as f:
        lines=f.readlines()

        for line in tqdm(lines[3:len(lines)],desc='loading object-tiept data'):
            line_split=str.split(line,sep='\t')
            if line_split[1] not in err_name:
                idce=np.where((tiept_info['object_name']==line_split[1]))[0]
                tiept_info['objpt_x'][idce]=eval(line_split[2])
                tiept_info['objpt_y'][idce]=eval(line_split[3])
                tiept_info['objpt_z'][idce]=eval(line_split[4])
                tiept_info['overlap'][idce]=eval(line_split[5])

def read_cp_file(cp_file):
    '''
    load control points file
    input:
    cp_file: the control point file
    output:
    cp: control points in dataframe format, [index	longitute	lattitute	level	height level	terrain slope]
    '''
    cp = pd.read_csv(cp_file, sep='\t', header=0)
    return cp

def numbering(points):
    ids=[]
    for point in points:
        if (-30<point[1]<60 and 27<point[0]<59.5) or (-10<point[1]<60 and 59.5<point[0]<80):
            ids.append(1)
        elif(-30<point[1]<60 and -50<point[0]<27):
            ids.append(2)
        elif(0<point[1]<80 and 60<point[0]<180) or (59.5<point[1]<80 and -180<point[0]<-170):
            ids.append(3)
        elif(-50<point[1]<0 and 60<point[0]<180):
            ids.append(4)
        elif(0<point[1]<59.5 and -180<point[0]<-30) or (59.5<point[1]<80 and -180<point[0]<-60):
            ids.append(5)
        elif(-50<point[1]<0 and -180<point[0]<-30):
            ids.append(6)
        elif(-50<point[1]<0 and 0<point[0]<110):
            ids.append(7)
        elif(-79<point[1]<-50 and 110<point[0]<180):
            ids.append(8)
        elif(-79<point[1]<-50 and -180<point[0]<-90):
            ids.append(9)
        elif(-79<point[1]<-50 and -90<point[0]<0):
            ids.append(10)
        elif(59.5<point[1]<80 and -10<point[0]<-60):
            ids.append(11)
    return set(ids)

def box_extract(cp_path,level_threshold,slope_threshold,box=None):
    nearby_cp_filtered=pd.DataFrame(columns=['n','lon','lat','h_interp','ac_level','terrain_slope'])
    if os.path.isdir(cp_path):
        cp_sub_dir=cp_path+'/data/'
        subdir_name=str.split(cp_path,sep='/')[-1]
        tmp_split=str.split(subdir_name,sep='_')
        tmp_points=np.array([[box[0],box[1]],[box[2],box[3]]])
        if eval(tmp_split[0]) in numbering(tmp_points):
            cp_files=os.listdir(cp_sub_dir)
            for cp_file in cp_files:
                tqdm.write(f'searching in the control point file {cp_file}...') 
                control_points=read_cp_file(os.path.join(cp_sub_dir,cp_file))
                indices=(box[0]<=control_points['lon']) & (control_points['lon']<=box[2]) & (box[1]<=control_points['lat']) & (control_points['lat']<=box[3]) & \
                (control_points['ac_level']<=level_threshold) & (control_points['terrain_slope'] <= slope_threshold)
                nearby_cp_filtered=pd.concat([nearby_cp_filtered,control_points[indices]],axis=0) 
    return nearby_cp_filtered

def sla_combine(search_range):
    tiept_info['is_sla']=np.zeros(len(tiept_info['objpt_z']))
    slapt_info_tree=neighbors.KDTree(slapt_info[['lon_prjd','lat_prjd']].values)
    # for idx in range(len(tiept_info)):
    #     nearby_cp_indices=slapt_info_tree.query_radius(tiept_info.loc[idx,['lon_prjd','lat_prjd']].values.reshape(1, -1),r=search_range,count_only = False, return_distance = False)
    nearby_cp_indices=slapt_info_tree.query_radius(tiept_info[['lon_prjd','lat_prjd']].values,r=search_range,count_only = False, return_distance = False)
    print(nearby_cp_indices)
    
    tiept_indices=[]
    for i in tqdm(range(len(nearby_cp_indices)),desc='finding nearby sla points'):
        if len(nearby_cp_indices[i])>0:
            tiept_indices.append(i)
    
    for idx in tqdm(tiept_indices,desc='combining'):
        sla_idx=nearby_cp_indices[idx][0].astype(np.int)
        tiept_info.loc[idx,'objpt_z']=slapt_info.loc[sla_idx,'h_interp']
        tiept_info.loc[idx,'is_sla']=1

def output_cps(output_file,output_type=None,method_type=None):
    if output_type=="SAR":
        if method_type=='adjacent':
            with open(output_file,'w+') as f:
                amount=len(tiept_info)
                f.write(str(amount)+'\n')
                for index in tqdm(range(amount),desc='writing to file'):
                    if slapt_info.loc[index,'nearby_cp'].size!=0:
                        f.write('Begin\t%s\t%.6f\t%.6f\t%.6f\n' %(str(tiept_info.loc[index,'pointID']),tiept_info.loc[index,'lon'],tiept_info.loc[index,'lat'],tiept_info.loc[index,'level']))
                        nearby_cp_tmp=slapt_info.loc[index,'nearby_cp']
                        for each in nearby_cp_tmp:
                            f.write('%.6f\t%.6f\t%.6f\n' %(each[0],each[1],each[2]))
                        f.write('End\n')   
                        f.write('\n')
        elif method_type=='box':
            np.savetxt(output_file, slapt_info, header='n lon lat h_interp ac_level terrain_slope', comments='', delimiter='\t', fmt='%d %f %f %f %d %f')
    elif output_type=='XQsoftware':
        with open(output_file,'w+') as f:
            amount=len(tiept_info)
            f.write("{}\n".format(amount))
            for idx in tqdm(tiept_info.index,desc='writing tiepoints'):
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:d}\t{:d}\n".format(tiept_info.loc[idx,'imgpt_id'],tiept_info.loc[idx,'object_name'],\
                    tiept_info.loc[idx,'img_id'],tiept_info.loc[idx,'objpt_x'],tiept_info.loc[idx,'objpt_y'],tiept_info.loc[idx,'objpt_z'],tiept_info.loc[idx,'imgpt_y'],tiept_info.loc[idx,'imgpt_x'],1,1))
        print('the result is written!')

def rewrite_order():
    with open(order_file,'r+') as f_r:
        lines=f_r.readlines()
    
    with open(order_file,'w+') as f_w:
        lines[8]="SLAFile: {}\n".format(tiesla_file)
        f_w.writelines(lines)

#%% input files 
if __name__=='__main__':
    order_file=r"F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\zdb.tri.txt"
    tiesla_file=r'/home/satis/zhaoyan/sla/shandong/tie.sla'
    err_tiept_file=r"F:\\phD_career\\multi_source_adjustment\\data\\guangzhou-demo\\auxiliary\\post.errtie.txt"
    objtiept_file=r"F:\phD_career\multi_source_adjustment\data\guangzhou-demo\auxiliary\ObjResidual.txt.tie"

#%% predefine parameters 
    max_core=cpu_count()

#%% analyse order file
    order_info,image_info=load_order_file(order_file)
    load_rpc_file(type='txt')
    err_name=load_errtiept_data(err_tiept_file)

    tiept_num,tiept_info=load_imgtiept_data(err_name,order_info['Tiefile'][0])
    load_objtiept_data(err_name,objtiept_file)
    tiept_info=pd.DataFrame(tiept_info)

    # tiept_info.to_csv(r'/home/satis/zhaoyan/173/guangzhou/tiept_const_hei.txt')
    # tiept_info=pd.read_csv(r'/home/satis/zhaoyan/173/guangzhou/tiept_const_hei.txt')
    # slapt_info=pd.read_csv(r'/home/satis/zhaoyan/173/guangzhou/temp_sla.txt')

#%% combining sla points
    sla_dirs=os.listdir(sla_root)
    sla_paths=[os.path.join(sla_root,each) for each in sla_dirs]
    boxes=[[tiept_info['objpt_x'].min(),tiept_info['objpt_y'].min(),tiept_info['objpt_x'].max(),tiept_info['objpt_y'].max()]]*len(sla_dirs)
    level_thresholds=[3]*len(sla_dirs)
    slope_thresholds=[0.1]*len(sla_dirs)
    paras=sla_paths,level_thresholds,slope_thresholds,boxes

    slapt_info=pd.DataFrame(columns=['n','lon','lat','h_interp','ac_level','terrain_slope'])
    with futures.ThreadPoolExecutor(max_workers=max_core) as texecutor:
        slapt_groups=texecutor.map(box_extract,*paras)

        for slapt_group in slapt_groups:
            slapt_info=pd.concat([slapt_info,slapt_group],axis=0)

    slapt_info.sort_values(by='n', inplace=True)
    slapt_info.index=range(len(slapt_info))

    geosrs="epsg:4326"
    prosrs="epsg:32651"
    transformer = pyproj.Transformer.from_crs(geosrs, prosrs)
    xprj,yprj=transformer.transform(tiept_info['objpt_y'],tiept_info['objpt_x'])
    tiept_info['lon_prjd']=xprj
    tiept_info['lat_prjd']=yprj
    p1 = pyproj.Proj(init=geosrs)
    p2 = pyproj.Proj(init=prosrs)
    xprj, yprj = pyproj.transform(p1, p2,x=slapt_info['lon'],y=slapt_info['lat'])
    slapt_info['lon_prjd']=xprj
    slapt_info['lat_prjd']=yprj

    sla_combine(50)
    tiept_info.to_csv(r'/home/satis/zhaoyan/sla/sla_combining/demos/temp_itersec.txt')
    tiept_info=tiept_info.loc[tiept_info['is_sla']==1]

#%% rewrite order file
    output_cps(tiesla_file,'XQsoftware')

    rewrite_order()
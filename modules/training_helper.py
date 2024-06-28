import torch
import numpy as np
import h5py
import pandas as pd
# from modules.plot_generator import plot_windy,plot_windy_5km
from modules.plot_generator_swin import plot_windy,plot_windy_5km


def output_sample_rainfall_chart(model, sample_data, summary_writer, epoch_index,plot_windy_size, device):
    model.eval()
    for phase, (sample_windy_EC, sample_rainfall, sample_time) in sample_data.items():
        sample_windy_EC, sample_rainfall = sample_windy_EC.to(device), sample_rainfall.to(device)
        pred_rainfall = model(sample_windy_EC)
        charts = []
        # Assuming sample_windy_EC and pred_rainfall are 4D tensors (B, C, H, W)
        sample_windy_EC = sample_windy_EC.mean(dim=1)
        pred_rainfall = pred_rainfall.mean(dim=1)
        sample_windy_EC = sample_windy_EC.cpu()
        pred_rainfall = pred_rainfall.cpu()
        sample_rainfall = sample_rainfall.cpu()
        if plot_windy_size=='5km':
            for i in range(2):
                charts.append(
                    plot_windy_5km(sample_windy_EC[i], pred_rainfall[i], sample_rainfall[i], sample_time[i])
                )
        elif plot_windy_size=='1.25km':
            for i in range(2):
                charts.append(
                    plot_windy(sample_windy_EC[i], pred_rainfall[i], sample_rainfall[i], sample_time[i])
                )

        chart_matrix = np.stack(charts).astype(np.int_)
        chart_matrix = chart_matrix[:,:,:,:-1]
        # with summary_writer.as_default():
        for case_i in range(chart_matrix.shape[0]):
            summary_writer.add_image(
                f'{phase}_rainfall_chart/{case_i}',
                chart_matrix[case_i,:,:,:].astype(np.float_)/255,
                epoch_index,
                # max_outputs=chart_matrix.shape[0],
                dataformats='HWC'
            )

        
def get_sample_data(dataset, count):
    for batch_index, (windy_EC, rainfall, time_train) in enumerate(dataset):
        valid_rain = rainfall[:, 0, 0] != -999
        sample_windy_EC = windy_EC[valid_rain][:count, ...]
        sample_rainfall = rainfall[valid_rain][:count, ...]
        sample_time = time_train[valid_rain][:count, ...]
        return sample_windy_EC, sample_rainfall, sample_time
    
def cost_loss_weighting_mse_hand(y_true, y_pred, loss_ratio):
    y_pred = torch.mean(y_pred, dim=1)
    w00 = torch.tensor((y_true < 2).float()) * loss_ratio.get('rain_gate_2', 0.0)
    w01 = torch.tensor(((y_true > 2) & (y_true <= 5)).float()) * loss_ratio.get('rain_gate_2_5', 0.0)
    w02 = torch.tensor(((y_true > 5) & (y_true <= 10)).float()) * loss_ratio.get('rain_gate_5_10', 0.0)
    w03 = torch.tensor(((y_true > 10) & (y_true <= 30)).float()) * loss_ratio.get('rain_gate_10_30', 0.0)
    w04 = torch.tensor((y_true >= 30).float()) * loss_ratio.get('rain_gate_30', 0.0)
    
    weights = w00 + w01 + w02 + w03 + w04
    cost = torch.nn.functional.mse_loss(y_pred, y_true, reduction='none') * weights
    cost = torch.mean(cost)
    
    return cost

def evaluate_loss(model, dataset, loss_ratio, device):
    # loss_basic = loss_basic = torch.nn.L1Loss()
    avg_rainfall_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (windy_EC, rainfall, times_train) in enumerate(dataset):
            windy_EC, rainfall = windy_EC.to(device), rainfall.to(device)
            pred_rainfall = model(windy_EC)
            batch_rainfall_loss = cost_loss_weighting_mse_hand(rainfall, pred_rainfall, loss_ratio)
            # batch_rainfall_loss = loss_basic(rainfall, torch.mean(pred_rainfall, dim=1)) 
            avg_rainfall_loss += batch_rainfall_loss.item()

    rainfall_loss = avg_rainfall_loss / len(dataset)

    return rainfall_loss

def get_torch_datasets(data_folder, batch_size, shuffle_buffer,data_channel):
    start_lat = 4
    end_lat = -4
    datasets = dict()
    print('data preparing...')
    with h5py.File(data_folder, 'r') as hf:
        combine_QPE_rain_3hr_read = hf['QPE_rain_3hr'][:]
        combine_QPE_rain_24hr_read = hf['QPE_rain_24hr_label'][:]
    datetime_old = pd.read_hdf(data_folder, keys='Time', mode='r')

    total_size =len(combine_QPE_rain_24hr_read[:,0,0])
    train_index = [x for x in range(total_size) if (x+1)%5 ==1 or (x+1)%5 ==2 or (x+1)%5 ==4]
    vaild_index = [x for x in range(total_size) if (x+1)%5 ==3]
    test_index  = [x for x in range(total_size) if (x+1)%5 ==0]
    
    if shuffle_buffer == '2022':
        total_size =len(combine_QPE_rain_24hr_read[:,0,0])
        train_index = [x for x in range(total_size) if (x+1)%5 ==0 or (x+1)%5 ==1 or (x+1)%5 ==2 or (x+1)%5 ==4]
        vaild_index = [x for x in range(total_size) if (x+1)%5 ==3]
        test_index  = [x for x in range(total_size) if (x+1)%5 ==0]
    

    if data_channel == 8:
        print('data_channel=8')
        data_train = combine_QPE_rain_3hr_read[train_index,:,start_lat:end_lat,:]
        data_label = combine_QPE_rain_24hr_read[train_index,:,start_lat:end_lat]
        data_time_train = [datetime_old.datetime_start_LST[i] for i in train_index]
        data_valid = combine_QPE_rain_3hr_read [vaild_index,:,start_lat:end_lat,:]
        data_valid_label = combine_QPE_rain_24hr_read [vaild_index,:,start_lat:end_lat]
        data_time_valid = [datetime_old.datetime_start_LST[i] for i in vaild_index]
        data_test = combine_QPE_rain_3hr_read [test_index,:,start_lat:end_lat,:]
        data_test_label = combine_QPE_rain_24hr_read [test_index,:,start_lat:end_lat]
        data_time_test = [datetime_old.datetime_start_LST[i] for i in test_index]
    
    elif data_channel==4:
        print('data_channel=4')
        data_train = np.zeros((len(train_index),112,104,4))
        for i in range(4):
            data_train[:,:,:,i] =np.sum(combine_QPE_rain_3hr_read[train_index,:,start_lat:end_lat,2*i:2*(i+1)],axis=3)
        data_label = combine_QPE_rain_24hr_read[train_index,:,start_lat:end_lat]
        data_time_train = [datetime_old.datetime_start_LST[i] for i in train_index]
        
        data_valid = np.zeros((len(vaild_index),112,104,4))
        for i in range(4):
            data_valid[:,:,:,i] =np.sum(combine_QPE_rain_3hr_read[vaild_index,:,start_lat:end_lat,2*i:2*(i+1)],axis=3)
        data_valid_label = combine_QPE_rain_24hr_read [vaild_index,:,start_lat:end_lat]
        data_time_valid = [datetime_old.datetime_start_LST[i] for i in vaild_index]
        
        data_test = np.zeros((len(test_index),112,104,4))
        for i in range(4):
            data_test[:,:,:,i] =np.sum(combine_QPE_rain_3hr_read[test_index,:,start_lat:end_lat,2*i:2*(i+1)],axis=3)
        data_test_label = combine_QPE_rain_24hr_read [test_index,:,start_lat:end_lat]
        data_time_test = [datetime_old.datetime_start_LST[i] for i in test_index]

    elif data_channel==1:
        print('data_channel=1')
        data_train = np.sum(combine_QPE_rain_3hr_read[train_index,:,start_lat:end_lat,:],axis=3, keepdims=True)
        data_label = combine_QPE_rain_24hr_read[train_index,:,start_lat:end_lat]
        data_time_train = [datetime_old.datetime_start_LST[i] for i in train_index]
        data_valid = np.sum(combine_QPE_rain_3hr_read [vaild_index,:,start_lat:end_lat,:],axis=3, keepdims=True)
        data_valid_label = combine_QPE_rain_24hr_read [vaild_index,:,start_lat:end_lat]
        data_time_valid = [datetime_old.datetime_start_LST[i] for i in vaild_index]
        data_test = np.sum(combine_QPE_rain_3hr_read [test_index,:,start_lat:end_lat,:],axis=3, keepdims=True)
        data_test_label = combine_QPE_rain_24hr_read [test_index,:,start_lat:end_lat]
        data_time_test = [datetime_old.datetime_start_LST[i] for i in test_index]
                

    data_train = np.swapaxes(np.swapaxes(data_train,2,3),1,2)
    data_valid = np.swapaxes(np.swapaxes(data_valid,2,3),1,2)
    data_test  = np.swapaxes(np.swapaxes(data_test,2,3),1,2)

    windy_EC_train = torch.from_numpy(data_train.astype('float32'))
    QPESUMS_train = torch.from_numpy(data_label.astype('float32'))
    times_train = torch.from_numpy(np.array(data_time_train).astype('int32'))

    windy_EC_valid = torch.from_numpy(data_valid.astype('float32'))
    QPESUMS_valid = torch.from_numpy(data_valid_label.astype('float32'))
    times_valid = torch.from_numpy(np.array(data_time_valid).astype('int32'))
    
    windy_EC_test = torch.from_numpy(data_test.astype('float32'))
    QPESUMS_test = torch.from_numpy(data_test_label.astype('float32'))
    times_test = torch.from_numpy(np.array(data_time_test).astype('int32'))

    train_dataset = torch.utils.data.TensorDataset(windy_EC_train, QPESUMS_train, times_train)
    datasets['train']  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(windy_EC_valid, QPESUMS_valid, times_valid)
    datasets['valid']  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(windy_EC_test, QPESUMS_test, times_test)
    datasets['test']  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return datasets

def get_torch_new_datasets(data_folder, batch_size, shuffle_buffer,data_channel):
    datasets = dict()
    f1 = h5py.File(data_folder+'2021/IFS_2021_1of2.h5')
    f2 = h5py.File(data_folder+'2021/IFS_2021_2of2.h5')
    f3 = h5py.File(data_folder+'2022/IFS_2022_1of2.h5')
    precipitation1 = f1['Variables']['precipitation'][:]
    precipitation2 = f2['Variables']['precipitation'][:]
    precipitation3 = f3['Variables']['precipitation'][:]
    target1 = f1['Variables']['target'][:]
    target2 = f2['Variables']['target'][:]
    target3 = f3['Variables']['target'][:]
    time1 = f1['Coordinates']['time'][:]
    time2 = f2['Coordinates']['time'][:]
    time3 = f3['Coordinates']['time'][:]
    
    data_train = np.expand_dims(np.concatenate([precipitation1,precipitation2, precipitation3],axis=0),axis=1)
    data_label = np.concatenate([target1,target2, target3],axis=0)
    train_time_tmp = np.concatenate([time1,time2, time3],axis=0)
    train_time = [np.int_(train_time_tmp[i][:4]+train_time_tmp[i][5:7]+train_time_tmp[i][8:10]+train_time_tmp[i][11:13]) for i in range(len(train_time_tmp))]
    
    # validation
    f = h5py.File(data_folder+'2022/IFS_2022_2of2.h5')
    data_valid = np.expand_dims(f['Variables']['precipitation'][:],axis=1)
    data_valid_label = f['Variables']['target'][:]
    valid_time_tmp = f['Coordinates']['time'][:]
    valid_time = [np.int_(valid_time_tmp[i][:4]+valid_time_tmp[i][5:7]+valid_time_tmp[i][8:10]+valid_time_tmp[i][11:13]) for i in range(len(valid_time_tmp))]
    
    
    windy_EC_train = torch.from_numpy(data_train.astype('float32'))
    QPESUMS_train = torch.from_numpy(data_label.astype('float32'))
    times_train = torch.from_numpy(np.array(train_time).astype('int32'))

    windy_EC_valid = torch.from_numpy(data_valid.astype('float32'))
    QPESUMS_valid = torch.from_numpy(data_valid_label.astype('float32'))
    times_valid = torch.from_numpy(np.array(valid_time).astype('int32'))
    
    train_dataset = torch.utils.data.TensorDataset(windy_EC_train, QPESUMS_train, times_train)
    datasets['train']  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(windy_EC_valid, QPESUMS_valid, times_valid)
    datasets['valid']  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return datasets

    

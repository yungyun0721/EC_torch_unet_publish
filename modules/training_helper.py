import torch
import numpy as np
import os
import netCDF4 as nc
from modules.plot_generator import plot_windy


def remove_outlier_and_nan(numpy_array, upper_bound=1000):
    numpy_array = np.nan_to_num(numpy_array, copy=False)
    # numpy_array[numpy_array > upper_bound] = 0
    return numpy_array

def output_sample_rainfall_chart(model, sample_data, summary_writer, epoch_index,plot_windy_size, device):
    model.eval()
    for phase, (sample_images, sample_rainfall, sample_time) in sample_data.items():
        sample_images, sample_rainfall = sample_images.to(device), sample_rainfall.to(device)
        pred_rainfall = model(sample_images)
        charts = []
        # Assuming sample_images and pred_rainfall are 4D tensors (B, C, H, W)
        sample_images = sample_images.cpu()
        pred_rainfall = pred_rainfall.cpu()
        sample_rainfall = sample_rainfall.cpu()
        for i in range(2):
            charts.append(
                plot_windy(sample_images[i], pred_rainfall[i], sample_rainfall[i], sample_time[i])
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
    for batch_index, (images, rainfall, time_train) in enumerate(dataset):
        valid_rain = rainfall[:, 0, 0, 0] != -999
        sample_images = images[valid_rain][:count, ...]
        sample_rainfall = rainfall[valid_rain][:count, ...]
        sample_time = time_train[valid_rain][:count, ...]
        return sample_images, sample_rainfall, sample_time
    
def cost_loss_weighting_mse_hand(y_true, y_pred, loss_ratio):
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
        for batch_index, (images, rainfall, times_train) in enumerate(dataset):
            images, rainfall = images.to(device), rainfall.to(device)
            pred_rainfall = model(images)
            batch_rainfall_loss = cost_loss_weighting_mse_hand(rainfall, pred_rainfall, loss_ratio)
            # batch_rainfall_loss = loss_basic(rainfall, torch.mean(pred_rainfall, dim=1)) 
            avg_rainfall_loss += batch_rainfall_loss.item()

    rainfall_loss = avg_rainfall_loss / len(dataset)

    return rainfall_loss

def get_torch_datasets(data_folder, data_time_arg, batch_size, data_channel):
    datasets = dict()
    print('data preparing...')

    # load and combine data
    combine_time_data  = []

    train_year = os.listdir(data_folder['input'])
    train_year.sort()
    for year_i in train_year:
        input_files = os.listdir(data_folder['input']+year_i)
        input_files.sort()
        for file in input_files:
            input_tmp = nc.Dataset(data_folder['input']+year_i+'/'+file)
            time_tmp = input_tmp.variables['time'][:]+np.int_(file[2:-3])*1000
            time_tmp = list(map(str, time_tmp))
            input_data = input_tmp.variables['Himawari_images'][:]
            label_tmp = nc.Dataset(data_folder['label']+year_i+'/'+file)
            label_data = label_tmp.variables['qperr'][:]
            if year_i == train_year[0] and file == input_files[0]:
                combine_input_data = input_data
                combine_label_data = label_data
            else:
                combine_input_data = np.concatenate((combine_input_data, input_data), axis=0)
                combine_label_data = np.concatenate((combine_label_data, label_data), axis=0)
            combine_time_data = combine_time_data+time_tmp
        print(f'finish {year_i}------------------')
    print('finish data loading-----------------')
    data_channel = [data_channel_index-1 for data_channel_index in data_channel]
    combine_input_data = remove_outlier_and_nan(combine_input_data)[:,:-1,:,:][:,:,:-1,:][:,:,:,data_channel]
    combine_label_data = remove_outlier_and_nan(combine_label_data)[:,:-1,:,:][:,:,:-1,:]
 
    # divide train, valid, test year    
    valid_back = [year_i[-2:] for year_i in data_time_arg['data_valid']]
    test_back  = [year_i[-2:] for year_i in data_time_arg['data_test']]
    valid_index = []
    test_index = []
    train_index = []
    for index_i in range(len(combine_time_data)):
        if combine_time_data[index_i][:2] in valid_back:
            valid_index = valid_index+[index_i]
        elif combine_time_data[index_i][:2] in test_back:
            test_index = test_index+[index_i]
        else:
            train_index = train_index+[index_i]
    
    print(f"valid year: {data_time_arg['data_valid']} numbers: {len(valid_index)}")
    print(f"test year: {data_time_arg['data_test']} numbers: {len(test_index)}")
    print(f"train year total numbers: {len(train_index)}")

    combine_time_data = list(map(float, combine_time_data))
    combine_time_data = list(map(int, combine_time_data))
    
    input_data_train = torch.from_numpy(np.swapaxes(np.swapaxes(combine_input_data[train_index,:,:,:],2,3),1,2).astype('float32'))
    input_data_valid = torch.from_numpy(np.swapaxes(np.swapaxes(combine_input_data[valid_index,:,:,:],2,3),1,2).astype('float32'))
    input_data_test  = torch.from_numpy(np.swapaxes(np.swapaxes(combine_input_data[test_index,:,:,:],2,3),1,2).astype('float32'))
    
    label_data_train = torch.from_numpy(np.swapaxes(np.swapaxes(combine_label_data[train_index,:,:,:],2,3),1,2).astype('float32'))
    label_data_valid = torch.from_numpy(np.swapaxes(np.swapaxes(combine_label_data[valid_index,:,:,:],2,3),1,2).astype('float32'))
    label_data_test  = torch.from_numpy(np.swapaxes(np.swapaxes(combine_label_data[test_index,:,:,:],2,3),1,2).astype('float32'))

    times_train = torch.from_numpy(np.array(combine_time_data)[train_index].astype('int32'))
    times_valid = torch.from_numpy(np.array(combine_time_data)[valid_index].astype('int32'))
    times_test = torch.from_numpy(np.array(combine_time_data)[test_index].astype('int32'))

    train_dataset = torch.utils.data.TensorDataset(input_data_train, label_data_train, times_train)
    datasets['train']  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(input_data_valid, label_data_valid, times_valid)
    datasets['valid']  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(input_data_test, label_data_test, times_test)
    datasets['test']  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return datasets
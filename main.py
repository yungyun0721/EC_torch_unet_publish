import argparse
import os
import importlib
import pandas as pd
from modules.rainfall_downscaling_trainer import train_rainfall_downscaling
from modules.training_helper import evaluate_loss, get_torch_datasets
from modules.experiment_helper import parse_experiment_settings

os.environ['CUDA_VISIBLE_DEVICES']="0"

def prepare_model_save_path(experiment_name, sub_exp_name):
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')

    saving_folder = 'saved_models/' + experiment_name
    if not os.path.isdir(saving_folder):
        os.mkdir(saving_folder)

    model_save_path = saving_folder + '/' + sub_exp_name
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    return model_save_path

def create_model_by_experiment_settings(experiment_settings):

    def create_model_instance(model_name):
        model_class = importlib.import_module(f'model_library.{model_name}').Model
        return model_class()
    
    rainfall_downscaling = create_model_instance(experiment_settings['rainfall_downscaling'])    # choose profiler in model_library
    
    # if load_from:
    #     rainfall_downscaling.load_weights(f'{load_from}')
    # loaded_model = UNet()
    # loaded_model.load_state_dict(torch.load('unet_model_weights.pth'))
    return rainfall_downscaling

# This function is faciliating creating model instance in jupiter notebook
def create_model_by_experiment_path_and_stage(experiment_path, sub_exp_name):
    sub_exp_settings = parse_experiment_settings(experiment_path, only_this_sub_exp=sub_exp_name)
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']

    model_save_path = prepare_model_save_path(experiment_name, sub_exp_name)
    model = create_model_by_experiment_settings(sub_exp_settings)
    model_save_path = model_save_path+'/rainfall_downscaling.pth'
    return model, model_save_path
    ## put the weight in to model (need import torch)
    # model.load_state_dict(torch.load(model_save_path))

def execute_sub_exp(sub_exp_settings, action, run_anyway):
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']
    log_path = f'logs/{experiment_name}/{sub_exp_name}'

    print(f'Executing sub-experiment: {sub_exp_name}')
    if not run_anyway and action == 'train' and os.path.isdir(log_path):
        print('Sub-experiment already done before, skipped ಠ_ಠ')
        return

    summary_writer_path = log_path
    model_save_path = prepare_model_save_path(experiment_name, sub_exp_name)
    datasets = get_torch_datasets(**sub_exp_settings['data'])

    print('starting building...')

    if action == 'train':
        model = create_model_by_experiment_settings(sub_exp_settings)
        
        print('finish model building...')

        train_rainfall_downscaling(
            model,#rainfall_downscaling
            datasets,
            summary_writer_path,
            model_save_path,
            **sub_exp_settings['train_rainfall']
        )

    elif action == 'evaluate':
        model = create_model_by_experiment_settings(sub_exp_settings, load_from=model_save_path)
        for phase in datasets:
            unet_loss = evaluate_loss(model, datasets[phase], loss_function='MAE')
            loss_to_save =[[sub_exp_name,phase,unet_loss.numpy()]]
            print(f'{phase} unet_loss: {unet_loss}')

def main(action, experiment_path, run_anyway):

    # parse yaml to get experiment settings
    experiment_list = parse_experiment_settings(experiment_path)
    print('finish setting....')

    for sub_exp_settings in experiment_list:
        execute_sub_exp(sub_exp_settings, action, run_anyway)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', help='name of the experiment setting, should match one of them file name in experiments folder')
    parser.add_argument('--action', help='(train/evaluate)', default='train')
    parser.add_argument('--omit_completed_sub_exp', action='store_true')
    args = parser.parse_args()
    main(args.action, args.experiment_path, (not args.omit_completed_sub_exp))
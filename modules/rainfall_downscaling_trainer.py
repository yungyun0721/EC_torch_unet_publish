import torch
from collections import defaultdict
from modules.training_helper import evaluate_loss, get_sample_data
from modules.training_helper import output_sample_rainfall_chart, cost_loss_weighting_mse_hand
from torch.utils.tensorboard import SummaryWriter
def train_rainfall_downscaling(
    rainfall_downscaling,
    datasets,
    summary_writer_path,
    saving_path,
    evaluate_freq,
    max_epoch,
    early_stop_tolerance=None,
    overfit_tolerance=None,
    loss_function='WMSE',
    loss_ratio={},
    optim_setting = {},
    plot_windy_size='5km'
):
    summary_writer = SummaryWriter(summary_writer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rainfall_downscaling = rainfall_downscaling.to(device)
    optimizer_DS_model = torch.optim.Adam(rainfall_downscaling.parameters(),lr=optim_setting['lr'], betas=(optim_setting['betas1'], optim_setting['betas2']), eps=optim_setting['eps'], amsgrad=False)
    # gpu computation reduce
    # grad_scaler = torch.cuda.amp.GradScaler() 
    
    # loss_basic = torch.nn.L1Loss()

    sample_data = {
        phase: get_sample_data(datasets[phase], 2)
        for phase in ['train', 'valid']
    }
    # use stack to keep track of validation loss and help early stopping
    valid_loss_stack = []
    avg_losses = defaultdict(lambda: torch.tensor(0.0).to(device))

    for epoch_index in range(1, max_epoch+1):
        print(f'Executing epoch #{epoch_index}')
        rainfall_downscaling.train()
        for batch_index, (images, rainfall, times_train) in enumerate(datasets['train']):
            images, rainfall = images.to(device), rainfall.to(device)
            ## train_step(images, rainfall)
            optimizer_DS_model.zero_grad()
            pred_rainfall = rainfall_downscaling(images)
            # rainfall_loss = loss_basic(rainfall, torch.mean(pred_rainfall, dim=1))
            rainfall_loss = cost_loss_weighting_mse_hand(rainfall, pred_rainfall, loss_ratio)
            rainfall_loss.backward()
            optimizer_DS_model.step()
            avg_losses[f'rainfall_{loss_function}_loss'] += rainfall_loss
            ## reduce gpu computation gradient
            # grad_scaler.scale(rainfall_loss).backward()
            # torch.nn.utils.clip_grad_norm_(rainfall_downscaling.parameters())
            # grad_scaler.step(optimizer_DS_model)
            # grad_scaler.update()
        for loss_name, avg_loss in avg_losses.items():
            summary_writer.add_scalar(loss_name, avg_loss.item(), epoch_index)
            
        avg_losses = defaultdict(lambda: torch.tensor(0.0).to(device))

        for name, param in rainfall_downscaling.named_parameters():
            if name in ['calibrate_factor', 'calibrate_constant']:
                summary_writer.add_scalar(name, param.item(), epoch_index)

        if (epoch_index) % evaluate_freq == 0:
            print(f'Completed {epoch_index} epochs, do some evaluation')
            # draw profile chart
            if epoch_index % 100 == 0:
                output_sample_rainfall_chart(rainfall_downscaling, sample_data, summary_writer, epoch_index, plot_windy_size, device)
            # calculate blending loss
            indicator_for_early_stop = {}
            if epoch_index % 500 == 0:
                torch.save(rainfall_downscaling.state_dict(), saving_path + '/rainfall_downscaling_'+str(epoch_index)+'.pth')
            if epoch_index>3000 and epoch_index % 200 == 0:
                torch.save(rainfall_downscaling.state_dict(), saving_path + '/rainfall_downscaling_'+str(epoch_index)+'.pth')
                
            for phase in ['train', 'valid']:
                rainfall_loss = evaluate_loss(rainfall_downscaling, datasets[phase], loss_ratio, device)
                for name, loss in [('rainfall', rainfall_loss)]:
                    summary_writer.add_scalar(f'[{phase}] images: {name}_loss', loss, epoch_index) 
                indicator_for_early_stop[phase] = rainfall_loss
            # save the best profiler and check for early stopping
            while valid_loss_stack and valid_loss_stack[-1] >= indicator_for_early_stop['valid']:
                valid_loss_stack.pop()
            if not valid_loss_stack:
                torch.save(rainfall_downscaling.state_dict(), saving_path + '/rainfall_downscaling.pth')
                print('Get the best validation performance so far! Saving the model.')
                output_sample_rainfall_chart(rainfall_downscaling, sample_data, summary_writer, epoch_index, plot_windy_size, device)
                
            elif early_stop_tolerance and len(valid_loss_stack) > early_stop_tolerance:
                print('Exceed the early stop tolerance, training procedure will end!')
                break
            elif overfit_tolerance and (indicator_for_early_stop['valid'] - indicator_for_early_stop['train']) >= overfit_tolerance:
                print('Exceed the overfit tolerance, training procedure will end!')
                # since valid loss is using blending, if train loss can beat valid loss,
                # that probably means profiler is already overfitting.
                break
            valid_loss_stack.append(indicator_for_early_stop['valid'])
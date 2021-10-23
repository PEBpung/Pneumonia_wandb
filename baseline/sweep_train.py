import time
import copy
import torch
import numpy as np
import gc
from sklearn.metrics import confusion_matrix

def train_model(dataloaders, dataset_sizes, num_iteration, net, criterion, optim, scheduler, device, classes_name, wandb, num_epoch):
    wandb.watch(net, criterion, log='all', log_freq=10)
    
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 100

    classes_name = classes_name

    for epoch in range(num_epoch):
        all_labels, all_preds, all_prob = [], [], []

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            loss_arr = []
            running_corrects = 0
            running_loss = 0.0

            #train dataset 로드하기
            for iteration_th, (inputs, labels) in enumerate(dataloaders[phase]): #iteration_th: 몇 번재 iteration 인지 알려 줌 "ex) batch_th=0 ← 첫 번째 batch 시작"
                
                ###########################GPU에 데이터 업로드##########################
                inputs = inputs.to(device) #image 데이터 GPU에 업로드
                labels = labels.to(device) #label 데이터 GPU에 업로드 // {labels: 0 → Normal, labels: 1 → Pneumonia} <== alb_data_load_classification.py 참고
                ########################################################################

                # backward pass ← zero the parameter gradients
                optim.zero_grad()

                with torch.set_grad_enabled(phase == "train"): # track history if only in train
                    outputs = net(inputs) #output 결과값은 softmax 입력 직전의 logit 값들
                    _, preds = torch.max(outputs, 1) #pred: 0 → Normal <== labels 참고
                    #preds2 = outputs.sigmoid() > 0.5
                    loss = criterion(outputs, labels) #criterion에 output이 들어가면 softmax 이 후의 확률 값으로 변하고, 변환된 확률 값과 label을 비교하여 loss 계산

                    loss_arr += [loss.item()] #Iteration 당 Loss 계산


                    if phase == "train":
                        loss.backward() #계산된 loss에 의해 backward (gradient) 계산
                        optim.step() #계산된 gradient를 참고하여 backpropagation으로 update
                        
                        wandb.log({"Train Iteration loss": np.mean(loss_arr), "custom_step" : iteration_th})
                        print("TRAIN: EPOCH %04d / %04d | ITERATION %04d / %04d | LOSS %.4f" %
                        (epoch+1, num_epoch, iteration_th, num_iteration['train'], np.mean(loss_arr)))

                    elif phase == 'val':
                        all_labels += labels.to("cpu")
                        all_preds += preds.to("cpu") # .detach()를 붙여야 될까?
                        all_prob.extend(outputs.to("cpu").detach().numpy())
                         

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step_ReduceLROnPlateau(np.mean(loss_arr)) #learning rate scheduler 실행
                #scheduler.step(np.mean(loss_arr)) #←← warm-up 사용하지 않을 시 learning rate scheduler 실행
                wandb.log({'train_epoch_loss': epoch_loss, 'Epoch Train ACC': epoch_acc, 'custom_step' : epoch})
            
            elif phase == 'val':
                wandb.log({'val_epoch_loss': epoch_loss, 'Epoch Val ACC': epoch_acc, 'custom_step' : epoch})
                

            print(f'Epoch {phase} Loss: {epoch_loss :>.4f} Acc: {epoch_acc:>.4f}')

            # deep copy the model
            if  epoch_loss < best_loss and phase == 'val':
                best_all_labels, best_all_preds, best_all_prob = [], [], []
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())

                best_all_labels = all_labels
                best_all_preds = all_preds
                best_all_prob = all_prob

            if epoch+1 == num_epoch and phase == 'val':
                # ROC
                wandb.log({'roc': wandb.plots.ROC(best_all_labels, best_all_prob, classes_name)})
                # Precision Recall
                wandb.log({'pr': wandb.plots.precision_recall(best_all_labels, best_all_prob, classes_name)})
                # Confusion Matrix
                wandb.sklearn.plot_confusion_matrix(best_all_labels, best_all_preds, labels=classes_name)

        gc.collect()
        torch.cuda.empty_cache()
        print()
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net
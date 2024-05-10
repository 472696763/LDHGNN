import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from rphgnn.utils.metrics_utils import LogitsBasedMetric
import math


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


class TorchTrainModel(nn.Module):
    def __init__(self, metrics_dict=None, learning_rate=None, scheduler_gamma=None) -> None:

        super().__init__()

        self.metrics_dict = metrics_dict
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.stop_training = False 

        use_float16 = False

        if use_float16:
            self.autocast_dtype = torch.float16
            self.scalar = torch.cuda.amp.GradScaler()
        else:
            self.autocast_dtype = torch.float32
            self.scalar = None

        self.optimizer = None

 
    def predict(self, data_loader, training=False):
        last_status = self.training
        if training:
            self.train()
        else:
            self.eval()

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                batch_y_pred_list = []
                for step, ((batch_x, batch_y),_) in enumerate(tqdm(data_loader)):
                    batch_logits = self(batch_x)                    
                    batch_y_pred = self.output_activation_func(batch_logits)

                    batch_y_pred_list.append(batch_y_pred.cpu())

        y_pred = torch.concat(batch_y_pred_list, dim=0)

        self.train(last_status)
        return y_pred
    


    def evaluate(self, data_loader, log_prefix):
        self.eval()
        

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                batch_y_pred_list = []
                batch_y_list = []
                losses_list = []
                for step, ((batch_x, batch_y), idx) in enumerate(tqdm(data_loader)):
                    batch_logits = self(batch_x)
                    
                    batch_losses = self.loss_func(batch_logits, batch_y)
                    
                    if self.multi_label:
                        batch_y_pred = (torch.sigmoid(batch_logits) > 0.5).float()
                    else:
                        batch_y_pred = torch.argmax(batch_logits, dim=-1)

                    if self.metrics_dict is not None:
                        for metric in self.metrics_dict.values():
                            if isinstance(metric, LogitsBasedMetric):
                                metric(batch_logits, batch_y)
                            else:
                                metric(batch_y_pred, batch_y)

                    losses_list.append(batch_losses.detach().cpu().numpy())
                    batch_y_pred_list.append(batch_y_pred.detach().cpu().numpy())
                    batch_y_list.append(batch_y.detach().cpu().numpy())

        losses = np.concatenate(losses_list, axis=0)
        loss = losses.mean()
        logs = {}


        if self.metrics_dict is not None:
            with torch.no_grad():        
                for metric_name, metric in self.metrics_dict.items():
                    logs["{}_{}".format(log_prefix, metric_name)] = metric.compute().item()
                    metric.reset()

        return logs




    def train_step(self, batch_data,epoch, pre_loss):
        return {}


  
    def train_epoch(self, epoch, train_data_loader, all_pre_loss):
        self.train()

        batch_results_dict = {}
        step_pbar = tqdm(train_data_loader)
        # for each batch
        for step, batch_data in enumerate(step_pbar):

            # get the loss of the previous epoch
            batch_pre_loss = all_pre_loss[batch_data[1]]

            # training with the current batch (model updating)
            batch_result, now_loss = self.train_step(batch_data[0], epoch, batch_pre_loss)

            # update the loss
            all_pre_loss[batch_data[1]] = now_loss
            with torch.no_grad():
                for key, value in batch_result.items():
                    if key not in batch_results_dict:
                        batch_results_dict[key] = []
                    batch_results_dict[key].append(value)
    
            step_pbar.set_postfix(
                {key: "{:.4f}".format(value.item()) for key, value in batch_result.items()}
            )


        if self.scheduler is not None:
            self.scheduler.step()
            # self.scheduler_a.step()
            print("current learning_rate: ", self.scheduler.get_last_lr())

        with torch.no_grad():
            logs = {
                key: torch.stack(value, dim=0).mean().item() for key, value in batch_results_dict.items() 
            }

        return logs,all_pre_loss




    def fit(self, train_data, 
            epochs, 
            validation_data, 
            validation_freq,
            callbacks=None,
            initial_epoch=0,
            len_y = None,
            ):
        
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            print("create optimizer ...")

            if self.scheduler_gamma is not None:
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.scheduler_gamma)
            else:
                self.scheduler = None

        if callbacks is None:
            callbacks = []
        
        for callback in callbacks:
            callback.model = self

        for callback in callbacks:
            callback.on_train_begin()

        # record the loss of each node for decrease calculation
        pre_loss = torch.full((len_y,), 0, dtype=torch.float32).cuda()

        # training
        for epoch in range(initial_epoch, epochs):
            logs = {"epoch": epoch}
            self.train()
            print("start epoch {}:".format(epoch))
            train_logs, pre_loss = self.train_epoch(epoch, train_data, pre_loss)

            logs = {
                **logs,
                **train_logs
            }

            # validation
            if (epoch + 1) % validation_freq == 0:
                self.eval()
                eval_start_time = time.time()
                validation_logs = self.evaluate(validation_data, log_prefix="val")
                logs = {
                    **logs,
                    **validation_logs
                }
                print("==== eval_time: ", time.time() - eval_start_time)


            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
            
            
            if (epoch + 1) % validation_freq == 0:
                # np_logs = {key: np.array(value) for key, value in logs.items()}
                print("epoch = {}\tlogs = {}".format(epoch, logs))

            if self.stop_training:
                print("early stop ...")
                break



class CommonTorchTrainModel(TorchTrainModel):
    def __init__(self, metrics_dict=None, multi_label=False, loss_func=None, learning_rate=None, scheduler_gamma=None, train_strategy="common", num_views=None, cl_rate=None) -> None:

        super().__init__(metrics_dict, learning_rate, scheduler_gamma)

        self.multi_label = multi_label
        self.train_strategy = train_strategy
        self.num_views = num_views
        self.cl_rate = cl_rate
        self.device = "cuda"
        self.preLoss = None

        if loss_func is not None:
            self.loss_func = loss_func
        else:
            if self.multi_label:
                self.loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
                self.output_activation_func = torch.nn.Sigmoid()
            else:
                self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
                self.output_activation_func = torch.nn.Softmax(dim=-1)
        self.optimizer = None

    
    def common_forward_and_compute_loss(self, batch_x, batch_y):

        logits = self(batch_x)
        losses = self.loss_func(logits, batch_y)
        loss = losses.mean()


        return logits, loss

    def cl_forward_and_compute_loss(self, batch_x, batch_y, batch_train_mask,epoch,pre_loss):

        # model output
        logits_list = [self(batch_x) for _ in range(self.num_views)]

        # compute loss of each nodes
        ce_loss_list = [self.loss_func(logits[batch_train_mask], batch_y[batch_train_mask])
                        for logits in logits_list]
        ce_loss = torch.stack(ce_loss_list, dim=0).sum(dim=0)

        # get loss at the previous epoch
        preloss = pre_loss[batch_train_mask]


        # convert loss-decrease into probability
        probabilities = F.softmax(preloss-ce_loss, dim=0)

        # get training size
        size = training_scheduler(0.5, epoch, 250, scheduler='linear')
        num_large_losses = int(len(ce_loss) * size)

        # select the training node based on the probability and size
        selected_idx = torch.multinomial(probabilities, num_large_losses, replacement=False)

        # compute loss for the selected training nodes
        ce_loss_list = [self.loss_func(logits[selected_idx], batch_y[selected_idx])
                         for logits in logits_list]
        ce_loss = torch.stack(ce_loss_list, dim=0).sum(dim=0)
        ce_loss = ce_loss.mean()
        loss = ce_loss
        pre_loss[selected_idx] = torch.stack(ce_loss_list, dim=0).sum(dim=0).detach()

        # return model output and loss
        return logits_list[0], loss, pre_loss



    def train_step(self, batch_data,epoch,pre_loss):

        # train_start_time = time.time()
        self.train()
        with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):

            if self.train_strategy == "common":
                batch_x, batch_y = batch_data
                logits, loss = self.common_forward_and_compute_loss(batch_x, batch_y)

            elif self.train_strategy == "cl":

                batch_x, batch_y, batch_train_mask = batch_data
                logits, loss , NonReduc_loss = self.cl_forward_and_compute_loss(batch_x, batch_y, batch_train_mask,epoch, pre_loss)

            else:
                raise Exception("not supported yet")
            
        # print("forward_time: ", time.time() - train_start_time)

        self.optimizer.zero_grad()
        if self.scalar is None:
            loss.backward()
            self.optimizer.step()        
        else:
            self.scalar.scale(loss).backward()
            self.scalar.step(self.optimizer)
            self.scalar.update()

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                if self.multi_label:
                    batch_y_pred = logits > 0.0
                else:
                    batch_y_pred = logits.argmax(dim=-1)
                    
                batch_corrects = (batch_y_pred == batch_y).float()
                batch_accuracy = batch_corrects.mean()

        return {
            "loss": loss, 
            "accuracy": batch_accuracy
        },NonReduc_loss


  
import os
import numpy as np
import json
import torch
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
from .constants import *
from pathlib import Path
import sklearn.metrics as metrics

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so that learning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7, verbose=verbose)
    return lr_scheduler





class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        task,
        criterion,
        optimizer,
        lr_scheduler,
        early_stopping_patience,
        device,
        parallel_devices,
        model_dir,
        model_name,
        considered_labels=[0, 1], target_names=None
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.task = task
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_step = 0
        self.early_stopping_break_signal = False
        self.device = device
        
        self.model_dir = model_dir
        self.model_name = model_name
        if self.task == 'clf':
            self.perfo = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []}
        elif self.task == 'reg':
            self.perfo = {"train_loss": [], "val_loss": [], "val_mse": [], "val_mae": [], "val_r2": [], "val_acc": []}
            
        self.global_step = 0
        self.summary_writer = SummaryWriter(model_dir)
        
        self.model.to(self.device)
        if parallel_devices:
            self.model = nn.DataParallel(self.model, device_ids=parallel_devices)
        
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
        self.considered_labels = considered_labels
        self.target_names = target_names
            

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            self._train_epoch()
            self._validate_epoch()
            if self.task == 'clf':
                print(
                    "Epoch: {}/{} || Train Loss={:.5f}, Val Loss={:.5f} || Train Acc={:.5f}, Val Acc={:.5f} || Val Prec={:.5f}, Val Rec={:.5f}, Val f1={:.5f}".format(
                        epoch + 1,
                        self.epochs,
                        self.perfo["train_loss"][-1],
                        self.perfo["val_loss"][-1],
                        self.perfo["train_acc"][-1],
                        self.perfo["val_acc"][-1],
                        self.perfo["val_precision"][-1],
                        self.perfo["val_recall"][-1],
                        self.perfo["val_f1"][-1],
                    )
                )
            elif self.task == 'reg':
                print(
                    "Epoch: {}/{} || Train Loss={:.5f}, Val Loss={:.5f} || Val MSE={:.5f}, Val MAE={:.5f}, Val R2={:.5f}  Val acc={:.5f}".format(
                        epoch + 1,
                        self.epochs,
                        self.perfo["train_loss"][-1],
                        self.perfo["val_loss"][-1],
                        self.perfo["val_mse"][-1],
                        self.perfo["val_mae"][-1],
                        self.perfo["val_r2"][-1],
                        self.perfo["val_acc"][-1],
                    )
                )
                
            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self.early_stopping_patience:
                self._check_early_stopping()
                if self.early_stopping_break_signal:
                    print("Early stopping signal: STOP")
                    break
                
            # Save checkpoint if best
            self._check_best_model(epoch)
            
            self.global_step += 1
        
        self.summary_writer.close()


    def _train_epoch(self):
        self.model.train()
        n_correct = 0
        n_samples = 0
        
        train_loss = 0
        for index, dataset in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), leave=False)):
            batch_input_ids = dataset['ids'].to(self.device, dtype = torch.long)
            batch_att_mask = dataset['att_mask'].to(self.device, dtype = torch.long)
            batch_target = dataset['target'].to(self.device, dtype = torch.long if self.task=='clf' else float)
                    
            output = self.model(batch_input_ids, 
                        token_type_ids=None,
                        attention_mask=batch_att_mask,
                        labels=batch_target if self.task=='clf' else None)
            
            step_loss = output.loss
            prediction = output.logits
            
            if self.task == 'reg':
                step_loss = self.criterion(prediction.squeeze(-1).float(), batch_target.squeeze(-1).float())
            
            step_loss.sum().backward()
            self.optimizer.step()        
            train_loss += step_loss
            self.optimizer.zero_grad()
            
            # Accuracy
            if self.task == 'clf':
                eval_prediction = np.argmax(prediction.detach().to('cpu').numpy(), axis=2 if len(prediction.shape) == 3 else 1)
                actual = batch_target.to('cpu').numpy()
                n_samples += len(batch_target.flatten())
                n_correct += (eval_prediction == actual).sum().item()
        
        epoch_loss = train_loss.sum().item()
        self.perfo["train_loss"].append(epoch_loss)
        
        if self.task == 'clf':
            acc = n_correct / n_samples if self.task == 'clf' else 0
            self.perfo["train_acc"].append(acc)
            self.summary_writer.add_scalar('train_acc', acc, global_step=self.global_step)
        
        self.summary_writer.add_scalar('train_loss', epoch_loss, global_step=self.global_step)
        self.summary_writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step=self.global_step)    
        


    def _validate_epoch(self):
        self.model.eval()

        eval_loss = 0
        
        predictions = []
        true_labels = []

        with torch.no_grad():
            for index, dataset in enumerate(self.val_dataloader):
                batch_input_ids = dataset['ids'].to(self.device, dtype = torch.long)
                batch_att_mask = dataset['att_mask'].to(self.device, dtype = torch.long)
                batch_target = dataset['target'].to(self.device, dtype = torch.long if self.task=='clf' else float)

                output = self.model(batch_input_ids, 
                            token_type_ids=None,
                            attention_mask=batch_att_mask,
                            labels=batch_target if self.task=='clf' else None)

                step_loss = output.loss
                eval_prediction = output.logits
                
                if self.task == 'reg':
                    step_loss = self.criterion(eval_prediction.squeeze(-1).float(), batch_target.squeeze(-1).float())

                eval_loss += step_loss
                
                # accuracy
                if self.task == 'clf':
                    if len(eval_prediction.shape) == 3:
                        eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis=2).flatten()
                        actual = batch_target.to('cpu').numpy().flatten()
                    else:
                        eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis=1)
                        actual = batch_target.to('cpu').numpy()
                        
                else:
                    lengths = batch_att_mask.sum(axis=1)
                    preds, targets = [], []
                    for b, length in enumerate(lengths):
                        preds.append(eval_prediction.squeeze(-1)[b][:length].detach().to('cpu').numpy())
                        targets.append(batch_target[b][:length].to('cpu').numpy())
                    eval_prediction = np.concatenate(preds)
                    actual = np.concatenate(targets)
                
                predictions.extend(eval_prediction)
                true_labels.extend(actual)


        epoch_loss = eval_loss.sum().item()
        self.perfo["val_loss"].append(epoch_loss)
        
        if self.task == 'clf':
            acc = metrics.accuracy_score(y_true=true_labels, y_pred=predictions)
            recall = metrics.recall_score(y_true=true_labels, y_pred=predictions, average='macro', labels=self.considered_labels)
            precision = metrics.precision_score(y_true=true_labels, y_pred=predictions, average='macro', labels=self.considered_labels)
            f1 = metrics.f1_score(y_true=true_labels, y_pred=predictions, average='macro', labels=self.considered_labels)
            print(metrics.classification_report(y_true=true_labels, y_pred=predictions, labels=self.considered_labels, target_names=self.target_names))
            
            self.perfo["val_acc"].append(acc)
            self.perfo["val_recall"].append(recall)
            self.perfo["val_precision"].append(precision)
            self.perfo["val_f1"].append(f1)
            
            self.summary_writer.add_scalar('eval_f1', f1, global_step=self.global_step)
            self.summary_writer.add_scalar('eval_acc', acc, global_step=self.global_step)
        else:
            mse = metrics.mean_squared_error(true_labels, predictions)
            mae = metrics.mean_absolute_error(true_labels, predictions)
            r2 = metrics.r2_score(true_labels, predictions)
            single_squared_errors = ((np.array(predictions) - np.array(true_labels)).flatten()**2).tolist()
            accuracy = sum([1 for e in single_squared_errors if e < 0.01]) / len(single_squared_errors)
            self.perfo["val_mse"].append(mse)
            self.perfo["val_mae"].append(mae)
            self.perfo["val_r2"].append(r2)
            self.perfo["val_acc"].append(accuracy)
        
            self.summary_writer.add_scalar('eval_mse', mse, global_step=self.global_step)
        
        self.summary_writer.add_scalar('eval_loss', epoch_loss, global_step=self.global_step)
        
        
    def _check_early_stopping(self):        
        valid_loss = self.perfo['val']
        if len(valid_loss) < self.early_stopping_patience:
            return

        # Check last validation losses are increasing
        if min(valid_loss) < valid_loss[-1]:
            self.early_stopping_step += 1
            print(f"Early stopping step : {self.early_stopping_step}/{self.early_stopping_patience}. Min eval_loss = {min(valid_loss):.5f}")
        else:
            self.early_stopping_step = 0
            
        self.early_stopping_break_signal = self.early_stopping_step >= self.early_stopping_patience

    
    def _check_best_model(self, epoch):
        if epoch == 0:
            return
        valid_loss = self.perfo['val_loss']
        min_loss_until = min(valid_loss[:-1])
        if valid_loss[-1] < min_loss_until:
            print(">>> Best model (loss)", end=" ")
            self._save_checkpoint(epoch, suffix="loss")
        
        if self.task == 'clf':
            valid_acc = self.perfo['val_acc']
            max_acc_until = max(valid_acc[:-1])
            if valid_acc[-1] > max_acc_until:
                print(">>> Best model (accuracy)", end=" ")
                self._save_checkpoint(epoch, suffix="acc")
                
            valid_f1 = self.perfo['val_f1']
            max_f1_until = max(valid_f1[:-1])
            if valid_f1[-1] > max_f1_until:
                print(">>> Best model (f1-score)", end=" ")
                self._save_checkpoint(epoch, suffix="f1")
            
    
    def _save_checkpoint(self, epoch, suffix=None):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if (epoch_num % self.checkpoint_frequency == 0) or suffix:
            if suffix:
                suffix = f"best_{suffix}"
                model_path = "{}.pt".format(suffix)
            else:    
                model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)
            print(f"Saving to {model_path}")


    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)


    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.perfo, fp)
            
    
        
            
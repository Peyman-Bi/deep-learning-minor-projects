import torch, os, pickle
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support
)

class TrainTest():
    def __init__(self, model, criterion, logger=None, writer=None):
        self.model = model
        self.criterion = criterion
        self.logger = logger
        self.writer = writer
        self.tr_metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'valid_loss': [],
            'valid_accuracy': [],
        }

    def train(
            self, train_loader, valid_loader,
            num_epochs, device, eval_interval,
            clip=None, model_path=None,
            save_per_epoch=None, results_path=None,
            model_name=''
    ):
        total_itrs = num_epochs * len(train_loader)
        num_tr, total_tr_loss, itr, step = 0, 0, 0, 0
        self.model.train()
        for epoch in range(num_epochs):
            for i, tr_inputs in enumerate(train_loader):
                self.optimizer.zero_grad()
                tr_output, tr_labels = self.feed_forward(tr_inputs, device)
                tr_loss = self.compute_loss(tr_output, tr_labels)
                # nn.utils.clip_grad_norm_(model.parameters(), clip)
                if self.logger:
                    self.logger.info(f'Training: {itr}/{total_itrs} -- loss: {tr_loss.item()}')
                tr_loss.backward()
                self.optimizer.step()
                num_tr += 1
                total_tr_loss += tr_loss
                if itr % eval_interval == 0 or itr + 1 == total_itrs:
                    self.tr_metrics['train_loss'].append(total_tr_loss.cpu().item() / num_tr)
                    tr_accuracy = self.cal_accuracy(tr_output, tr_labels)
                    self.tr_metrics['train_accuracy'].append(tr_accuracy)
                    num_tr, total_tr_loss = 0, 0
                    val_loss = 0
                    self.model.eval()
                    val_accuracy = []
                    with torch.no_grad():
                        for i, val_inputs in enumerate(valid_loader):
                            self.optimizer.zero_grad()
                            val_output, val_labels = self.feed_forward(val_inputs, device)
                            val_loss += self.compute_loss(val_output, val_labels)
                            val_accuracy.append(self.cal_accuracy(val_output, val_labels))
                    self.tr_metrics['valid_accuracy'].append(np.mean(val_accuracy))
                    self.tr_metrics['valid_loss'].append(val_loss.cpu().item() / len(valid_loader))
                    self.model.train()
                    if self.logger:
                        self.logger.info(f'Training: iteration: {itr}/{total_itrs} -- epoch: {epoch} -- '
                                         f' train_loss: {self.tr_metrics["train_loss"][-1]:.3f} -- train_accuracy: {self.tr_metrics["train_accuracy"][-1]:.2f}'
                                         f' valid_loss: {self.tr_metrics["valid_loss"][-1]:.3f} -- valid_accuracy: {self.tr_metrics["valid_accuracy"][-1]:.2f}')
                        self.writer.add_scalar('train-loss', self.tr_metrics["train_loss"][-1], global_step=step)
                        self.writer.add_scalar('valid-loss', self.tr_metrics["valid_loss"][-1], global_step=step)
                        self.writer.add_scalar('train-accuracy', self.tr_metrics["train_accuracy"][-1], global_step=step)
                        self.writer.add_scalar('valid-accuracy', self.tr_metrics["valid_accuracy"][-1], global_step=step)
                    step += 1
                itr += 1
            self.scheduler.step()
            if model_path and results_path and ((epoch + 1) % save_per_epoch == 0) and epoch != 0:
                self.save_model(epoch + 1, model_path, f'{model_name}_{epoch + 1}_epochs_train')
                self.save_results(results_path, f'{model_name}_{epoch + 1}_epochs_train', self.tr_metrics)
        if model_path and results_path:
            self.save_model(epoch + 1, model_path, f'{model_name}_{epoch + 1}_epochs_last_train')
            self.save_results(results_path, f'{model_name}_{epoch + 1}_epochs_last_train', self.tr_metrics)

    def load_trained_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

    def load_training_results(self, path):
        with open(path, 'rb') as results_file:
            training_results = pickle.load(results_file)
        self.tr_metrics = training_results

    def continue_train(
            self, train_loader, valid_loader,
            model_path, results_path,
            num_epochs, device, eval_interval,
            clip=None, save_per_epoch=None, model_name=''
    ):
        last_epoch = self.load_trained_model(model_path)
        self.load_training_results(results_path)
        self.train(
            train_loader, valid_loader, num_epochs,
            device, eval_interval,
            model_path='/'.join([str(x) for x in model_path.split('/')[:-1]] ) +'/',
            save_per_epoch=save_per_epoch,
            results_path='/'.join([str(x) for x in results_path.split('/')[:-1]] ) +'/',
            clip=clip,
            model_name=model_name
        )

    def set_optimizer(self, optimizer, **kwargs):
        if optimizer == 'Adam':
            from torch.optim import Adam
            self.optimizer = Adam(self.model.parameters(), **kwargs)
        elif optimizer == 'SGD':
            from torch.optim import SGD
            self.optimizer = SGD(self.model.parameters(), **kwargs)
        else:
            self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def set_scheduler(self, scheduler, **kwargs):
        if scheduler == 'LambdaLR':
            from torch.optim.lr_scheduler import LambdaLR
            self.scheduler = LambdaLR(self.optimizer, **kwargs)
        if scheduler == 'StepLR':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(self.optimizer, **kwargs)
        else:
            self.scheduler = scheduler(self.optimizer, **kwargs)

    # this method should be inherited by the childerens
    def feed_forward(self, inputs, device):
        pass

    # this method might be inherited by the childerens
    def compute_loss(self, output, labels):
        return self.criterion(output, labels.view(-1))

    # this method might be inherited by the childerens
    @classmethod
    def cal_accuracy(cls, pred_labels, true_labels):
        _, pred_labels = pred_labels.max(dim=1)
        true_labels = true_labels.view(-1)
        return torch.sum(pred_labels == true_labels).item() / true_labels.size()[0]

    def save_model(self, epoch, model_path, name):
        model_dir = '/'.join(model_path.split('/')[:-1])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.tr_metrics['valid_loss'][-1],
            }, os.path.join(model_dir, f'model_{name}.pt')
        )
        if self.logger:
            self.logger.info(f'Training: model saved to: {model_dir}/model_{name}.pt')

    def save_results(self, results_path, name, results):
        results_dir = '/'.join(results_path.split('/')[:-1])
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(os.path.join(results_dir, f'results_{name}.pkl'), 'wb') as save_file:
            pickle.dump(results, save_file)
        if self.logger:
            self.logger.info(f'Training: results saved to: {results_dir}/resutls_{name}.pkl')

    def freeze_layers(self, freezing_param_names):
        for name, param in self.model.named_parameters():
            if name in freezing_param_names:
                param.requires_grad = False

    def test(self, ts_loader, device, model_name='', label_names=None, results_path=None):
        ts_accuracy, ts_true, ts_pred = [], [], []
        ts_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, ts_inputs in enumerate(ts_loader):
                self.optimizer.zero_grad()
                ts_output, ts_labels = self.feed_forward(ts_inputs, device)
                ts_loss += self.compute_loss(ts_output, ts_labels)
                ts_accuracy.append(self.cal_accuracy(ts_output, ts_labels))
                ts_true.append(ts_labels.cpu())
                ts_pred.append(ts_output.cpu().max(dim=1)[1])
        ts_true = torch.cat(ts_true)
        ts_pred = torch.cat(ts_pred)
        ts_loss = ts_loss.cpu().item() / len(ts_loader)
        ts_accuracy = np.mean(ts_accuracy)
        prf = precision_recall_fscore_support(
            ts_true,
            ts_pred,
            labels=label_names,
            average='weighted'
        )
        confm = confusion_matrix(ts_true, ts_pred, labels=label_names)
        self.ts_metrics = {
            'loss': ts_loss,
            'accuracy': ts_accuracy,
            'precision': prf[0],
            'recall': prf[1],
            'f1_score': prf[2],
            'confusion_matrix': confm
        }
        if self.logger:
            print(
                f'tsing: ts_loss: {ts_loss:.3f} -- ts_accurcy: {ts_accuracy:.2f}')
        if results_path:
            self.save_results(results_path, f'{model_name}_ts', self.ts_metrics)
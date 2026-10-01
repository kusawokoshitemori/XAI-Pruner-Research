import math
import sys
from typing import Iterable, Optional
import torch

from model import VisionTransformer
from pruner import create_crntroller, update_scores, get_pruned_model, save_state
from lrp import ViTLRP

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import engine.utils as utils
import random

import diag



def train_one_epoch(model: torch.nn.Module,
                    criterion,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    amp: bool = True,
                    teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None):
    model.train(True)
    criterion.train()

    random.seed(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_scores(model_lrp,
                   model_without_ddp,
                   print_freq,
                   criterion,
                   data_loader: Iterable,
                   optimizer: torch.optim.Optimizer,
                   momentum:float,
                   device: torch.device):


    model_lrp.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Computing Layer-wise Relevance'

    scores = None
    diag.set_phase("compute_scores")
    if diag.enabled():
        training_modules = [n for n, m in model_without_ddp.named_modules() if m.training]
        diag.emit("lrp_model_mode", training_module_count=len(training_modules), examples=training_modules[:5],
                  note="Filter の入力を次の Linear の分母とみなす対応は eval（Dropout/DropPath が恒等）が前提")

    for batch_id, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        diag.begin_batch(batch_id)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model_lrp(samples)
        loss = criterion(outputs, targets)

        relevance = torch.softmax(outputs, dim=1)
        max_indices = torch.argmax(relevance, dim=1, keepdim=True)
        mask = torch.zeros_like(relevance, dtype=torch.bool).scatter_(1, max_indices, True)
        relevance = torch.where(mask, relevance, torch.tensor(0))

        loss_value = loss.item()
        diag.scores.initial_relevance(batch_id, samples, outputs, targets, relevance, momentum)


        outputs.backward(relevance)

        optimizer.zero_grad()

        scores = update_scores(model_without_ddp, scores, momentum)
        diag.end_batch()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    diag.scores.finalize(scores, momentum)
    return scores

def prune_one_shot(
        model_lrp,
        model_without_ddp,
        model,
        print_freq,
        criterion,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        pruning_rate,
        population,
        protect_rate,
        epsilon,
        generation,
        save_dir,
        momentum,
        num_classes):

    scores = compute_scores(model_lrp=model_lrp, model_without_ddp=model_without_ddp, print_freq=print_freq, criterion=criterion,
                            data_loader=data_loader, optimizer=optimizer, device=device, momentum=momentum)

    model.to('cpu')

    pruner = create_crntroller(model=model, scores=scores, pruning_rate=pruning_rate,
                 population=population, epsilon=epsilon,protect=protect_rate, epochs=generation)

    diag.set_phase("evolution")
    pruner.engine(mutation_num=int(pruner.population/2), crossover_num=int(pruner.population /2))

    masks = pruner.get_mask()

    diag.set_phase("build_pruned_model")
    kind = "vit" if isinstance(model, VisionTransformer) else type(model).__name__.lower()
    try:
        pruned_model = get_pruned_model(model, masks, save_dir, num_classes)
    except Exception as e:
        diag.ea.model_built(model, None, kind, pruner, error=e)
        raise
    diag.ea.model_built(model, pruned_model, kind, pruner)

    return pruned_model




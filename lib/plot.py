import json
import matplotlib.pyplot as plt
import os


def read_log_file(log_file:str):
    epochs = []
    train_losses = []
    test_losses = []
    test_acc1 = []
    test_acc5 = []

    with open(log_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            epochs.append(data['epoch'])
            train_losses.append(data['train_loss'])
            test_losses.append(data['test_loss'])
            test_acc1.append(data['test_acc1'])
            test_acc5.append(data['test_acc5'])

    return epochs, train_losses, test_losses, test_acc1, test_acc5


def draw_loss_plot(epochs, train_losses, test_losses, save_dir):
    plt.figure(figsize=(10, 6), dpi=400)
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2, color='#41525a')
    plt.plot(epochs, test_losses, label='Basae-2 Loss', linewidth=2, color='#eb1c22')

    save_file = 'loss_plot.png'
    save_path = os.path.join(save_dir, save_file)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig(save_path)


def draw_acc_plot(epochs, test_acc1, test_acc5, save_dir):
    plt.figure(figsize=(10, 6), dpi=400)
    plt.plot(epochs, test_acc1, label='test_acc1', linewidth=2, color='#2fc0d3')
    plt.plot(epochs, test_acc5, label='test_acc5', linewidth=2, color='#066da1')

    save_file = 'acc_plot.png'
    save_path = os.path.join(save_dir, save_file)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy over Epochs')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig(save_path)



def plot(log_file, save_dir):
    epochs, train_losses, test_losses, test_acc1, test_acc5 = read_log_file(log_file)
    draw_loss_plot(epochs, train_losses, test_losses, save_dir)
    draw_acc_plot(epochs, test_acc1, test_acc5, save_dir)


if __name__ == '__main__':
    plot("/home/buenos/Desktop/PBL/Small/log.txt", "/home/buenos/Desktop/PBL/Small")
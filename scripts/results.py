import matplotlib.pyplot as plt
import pandas as pd

from myconfig import PATH_ACCURACY, PATH_LOSS, PATH_HISTORY

def plot_history(df):
    # Accuracy
    plt.figure(figsize = (10, 10))
    plt.plot(df['accuracy'], '-o', alpha = 0.8)
    plt.plot(df['val_accuracy'], '-o', alpha = 0.8)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.savefig(PATH_ACCURACY, bbox_inches = 'tight')

    # Loss
    plt.figure(figsize = (10, 10))
    plt.plot(df['loss'], '-o', alpha = 0.8)
    plt.plot(df['val_loss'], '-o', alpha = 0.8)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.savefig(PATH_LOSS, bbox_inches = 'tight')
    return

def main():
    df = pd.read_csv(PATH_HISTORY)
    plot_history(df)
    return

if __name__ == "__main__":
    main()

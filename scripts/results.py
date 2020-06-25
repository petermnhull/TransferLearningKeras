import matplotlib.pyplot as plt
import pandas as pd

def plot_history(df):
    # Accuracy
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.savefig('accuracy.png', bbox_inches = 'tight')

    # Loss
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper left')
    plt.savefig('loss.png', bbox_inches = 'tight')
    return

def main():
    df = pd.read_csv("history.csv")
    plot_history(df)
    return

if __name__ == "__main__":
    main()

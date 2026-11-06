import json
import matplotlib.pyplot as plt

# Sostituisci con il nome reale del tuo file JSON
JSON_FILE = 'training_history.json' 

def plot_history(json_path):
    # 1. Carica i dati dal file JSON
    with open(json_path, 'r') as f:
        history_dict = json.load(f)

    # Estrai le metriche
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(acc) + 1)

    # 2. Crea la figura con due grafici (uno per Accuracy, uno per Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico Accuracy
    ax1.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Grafico Loss
    ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Mostra i grafici
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_history(JSON_FILE)
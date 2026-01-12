import json

func_code = '''
def analyze_errors(y_true, y_pred, idx_to_class, test_gen, test_dir, img_size):
    """
    Analyse les erreurs du modèle : accuracy par classe, top confusions, et grille d'images mal classées.
    
    Args:
        y_true (np.ndarray): Labels réels.
        y_pred (np.ndarray): Labels prédits.
        idx_to_class (dict[int,str]): Mapping index -> nom de classe.
        test_gen: Générateur de test.
        test_dir (str): Chemin du dossier test.
        img_size (tuple): Taille des images.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from sklearn.metrics import confusion_matrix
    
    Path('figs').mkdir(parents=True, exist_ok=True)
    
    labels_all = list(range(len(idx_to_class)))
    class_names = [idx_to_class[i] for i in labels_all]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    
    # 1) Per-class accuracy
    row_sums = cm.sum(axis=1)
    per_class_acc = np.divide(cm.diagonal(), row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums > 0)
    per_class_df = pd.DataFrame({'classe': class_names, 'accuracy': per_class_acc}).sort_values(by='accuracy', ascending=True)
    
    plt.figure(figsize=(9, 4))
    sns.barplot(data=per_class_df, x='classe', y='accuracy', color='#4E79A7')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20)
    plt.ylabel('Accuracy')
    plt.title('Accuracy par classe (test)')
    plt.tight_layout()
    out_acc = 'figs/analysis_per_class_accuracy.png'
    plt.savefig(out_acc, dpi=200, bbox_inches='tight')
    print('Saved:', out_acc)
    plt.show()
    
    # 2) Top confusions
    pairs = []
    for i in range(len(labels_all)):
        for j in range(len(labels_all)):
            if i != j and cm[i, j] > 0:
                pairs.append((idx_to_class[i], idx_to_class[j], int(cm[i, j])))
    
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    if pairs_sorted:
        df_conf = pd.DataFrame(pairs_sorted, columns=['true', 'pred', 'count'])
        df_conf['pair'] = df_conf['true'] + ' → ' + df_conf['pred']
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_conf, y='pair', x='count', color='#F28E2B')
        plt.xlabel('Nombre de confusions')
        plt.ylabel('Paire (Vrai → Prédit)')
        plt.title('Top confusions')
        plt.tight_layout()
        out_conf = 'figs/analysis_top_confusions.png'
        plt.savefig(out_conf, dpi=200, bbox_inches='tight')
        print('Saved:', out_conf)
        plt.show()
    else:
        print('Aucune confusion à tracer.')
    
    # 3) Grille d'images mal classées
    mis_idx = np.where(y_true != y_pred)[0][:16]
    if len(mis_idx) > 0:
        cols, rows = 8, 2
        plt.figure(figsize=(2*cols, 2*rows))
        for k, idx in enumerate(mis_idx[:cols*rows]):
            path = os.path.join(test_dir, test_gen.filenames[idx])
            img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            plt.subplot(rows, cols, k+1)
            plt.imshow(arr)
            plt.title(f"V:{idx_to_class[y_true[idx]]}/P:{idx_to_class[y_pred[idx]]}", fontsize=8)
            plt.axis('off')
        plt.tight_layout()
        mis_path = 'figs/analysis_mispredictions_grid.png'
        plt.savefig(mis_path, dpi=200, bbox_inches='tight')
        print('Saved:', mis_path)
        plt.show()
    else:
        print('Aucune erreur à afficher.')
'''

with open('complet.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == '#VSC-36c99a55':
        cell['source'].insert(0, func_code)
        break

with open('complet.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
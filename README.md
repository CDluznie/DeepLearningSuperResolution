# Deep Learning Super-Resolution
L’objectif de ce projet est de mettre au point un réseau de neurones profonds pour la super-résolution d'image

## Dépendances
* TensorFlow
* NumPy
* SciPy

## Utilisation

* **Entrainement** : train.py
  * --model *model* : choix du modèle, valeurs possible : espcn, edspcn, edspcn par défaut
  * --dataset *dataset* : jeu de données d'entrainement, data/General-100 par défaut
  * --batchsize *batchsize* : nombre d'images par epoque, 20 par défaut
  * --epochs e : nombre d'iterations totale, 1000 par défaut

* **Super-Resolution** : upscale.py
  * --model *model* : choix du modèle, valeurs possible : espcn, edspcn, edspcn par défaut
  * --image *image* : image d'entrée, argument requis
    
* **Résumé de la phase d'entrainement** : tensorboard --logdir models/save/*MODEL*/train

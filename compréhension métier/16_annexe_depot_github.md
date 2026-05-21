# Annexe A : Code source et reproductibilité

L'intégralité du pipeline (60 scripts Python, soit 10 383 lignes de code) ainsi que la documentation méthodologique étendue sont disponibles sur le dépôt public suivant :

**[https://github.com/BD56/prediction-ffe-2026](https://github.com/BD56/prediction-ffe-2026)**

Le dépôt contient :

- **`scripts/`** : pipeline complet, de la construction des features (familles F1 à F10) à la calibration de l'intervalle de confiance LAC, en passant par les comparatifs de modèles (ElasticNet, XGBoost, CatBoost, Random Forest, Stacking) et les protocoles de validation multi-splits.
- **`compréhension métier/`** : documentation méthodologique détaillée (plan d'engineering, catalogue exhaustif des 273 features candidates, sources de littérature, limites discutées, validations).
- **`data/master/master_dataset_synthetic.parquet`** : jeu de 20 chevaux fictifs respectant le schéma complet, permettant l'exécution du pipeline sans accès aux données FFE.
- **`Fiche FFE.command`** : lanceur natif macOS de l'application de prédiction interactive.
- **`README.md`** et **`LICENSE`** (MIT) : présentation du projet et conditions de réutilisation du code.

Les **données FFE originales** ainsi que les résultats numériques publiés dans ce rapport ne sont pas redistribuables et restent soumis à l'accord de confidentialité du commanditaire. L'exécution du pipeline contre les vraies données nécessite un accès accordé directement par la FFE.

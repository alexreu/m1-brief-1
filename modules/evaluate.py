from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_performance(y_true, y_pred):
    """
    Fonction pour mesurer les performances du modèle avec MSE, MAE et R².

    MSE: mesurer l'erreur entre des valeurs prédites et des valeurs réelles.
    Faible le modèle prédit bien, élevé les prédictions sont loins des valeurs réelles

    MAE: indique de combien, en moyenne, les prédictions s’écartent des valeurs réelles

    R²: indique à quel point les prédictions du modèle correspondent aux données réelles
    1 = parfait / 0.5 prédiction moyenne / 0 mauvaise prédiction
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "R²": r2}

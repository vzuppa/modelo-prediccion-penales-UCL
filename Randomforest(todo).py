import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train = pd.read_excel("PENALESUCL.xlsx")
test = pd.read_excel("Penalestesting.xlsx")

variables_predictoras = [
    "Pie",
    "Movimiento_brazo",
    "Contexto",
    "Tiempo_de_ejecución",
    "Distancia_al_punto_penal",
    "Movimientos_arquero"
]

variable_objetivo = "Dirección"

variables_categoricas = [
    "Pie",
    "Movimiento_brazo",
    "Movimientos_arquero"
]

variables_numericas = [
    "Contexto",
    "Tiempo_de_ejecución",
    "Distancia_al_punto_penal"
]

X_train = train[variables_predictoras]
y_train = train[variable_objetivo]

X_test = test[variables_predictoras]
y_test = test[variable_objetivo]

preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            variables_categoricas
        ),
        (
            "numericas",
            "passthrough",
            variables_numericas
        )
    ]
)

modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesamiento),
        ("clasificador", RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, predicciones))

print("\nClases:")
print(modelo.classes_)

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, predicciones, labels=modelo.classes_))

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))

resultado = test.copy()
resultado["Predicción"] = predicciones

for i, clase in enumerate(modelo.classes_):
    resultado[f"Probabilidad_{clase}"] = probabilidades[:, i]

resultado.to_excel("predicciones_penales_modelo_completo.xlsx", index=False)

print("\nArchivo generado: predicciones_penales_modelo_completo.xlsx")
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

train = pd.read_excel("PENALESUCL.xlsx")
test = pd.read_excel("Penalestesting.xlsx")

variables_predictoras = ["Pie", "Movimiento_brazo"]
variable_objetivo = "Dirección"

X_train = train[variables_predictoras]
y_train = train[variable_objetivo]

X_test = test[variables_predictoras]
y_test = test[variable_objetivo]

datos_train = pd.concat([X_train, y_train], axis=1).dropna()
datos_test = pd.concat([X_test, y_test], axis=1).dropna()

X_train = datos_train[variables_predictoras]
y_train = datos_train[variable_objetivo]

X_test = datos_test[variables_predictoras]
y_test = datos_test[variable_objetivo]

preprocesamiento = ColumnTransformer(
    transformers=[
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), variables_predictoras)
    ]
)

modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesamiento),
        ("clasificador", RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, predicciones))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, predicciones, labels=modelo.classes_))

print("\nClases:")
print(modelo.classes_)

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))

resultado = datos_test.copy()
resultado["Predicción"] = predicciones

for i, clase in enumerate(modelo.classes_):
    resultado[f"Probabilidad_{clase}"] = probabilidades[:, i]

resultado.to_excel("predicciones_penales.xlsx", index=False)


"""
modelos = {
    "Regresión logística multinomial": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Árbol de decisión": DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        random_state=42,
        class_weight="balanced"
    )
}

for nombre, clasificador in modelos.items():
    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("clasificador", clasificador)
        ]
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print(nombre)
    print("Accuracy en testeo:", accuracy)
    print()
"""
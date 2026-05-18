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

print("=" * 50)
print("EVALUACIÓN DEL MODELO")
print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, predicciones):.4f}")
print(f"\nClases   : {list(modelo.classes_)}")
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, predicciones, labels=modelo.classes_))
print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))

OPCIONES_PIE   = ["Derecho", "Izquierdo"]
OPCIONES_BRAZO = ["EXTENDIDO", "ENCOGE"]
 
def elegir_opcion(mensaje, opciones):
    """Muestra las opciones numeradas y devuelve el valor elegido."""
    while True:
        print(f"\n{mensaje}")
        for i, op in enumerate(opciones, 1):
            print(f"  {i}. {op}")
        entrada = input("Ingresá el número: ").strip()
        if entrada.isdigit() and 1 <= int(entrada) <= len(opciones):
            return opciones[int(entrada) - 1]
        print("  ✗ Opción inválida, intentá de nuevo.")
 
def predecir_penal():
    print("\n" + "=" * 50)
    print("PREDICTOR DE PENALES")
    print("=" * 50)
 
    pie   = elegir_opcion("¿Cuál es el pie del pateador?", OPCIONES_PIE)
    brazo = elegir_opcion("¿Cuál es el movimiento del brazo?", OPCIONES_BRAZO)
 
    nuevo = pd.DataFrame({"Pie": [pie], "Movimiento_brazo": [brazo]})
    pred  = modelo.predict(nuevo)[0]
    proba = modelo.predict_proba(nuevo)[0]
 
    print("\n── Resultado ──────────────────────────────")
    print(f"  Pie             : {pie}")
    print(f"  Movimiento brazo: {brazo}")
    print(f"\n  Predicción      : {pred}")
    print("\n  Probabilidades:")
    for clase, prob in sorted(zip(modelo.classes_, proba), key=lambda x: -x[1]):
        barra = "█" * int(prob * 30)
        print(f"    {clase:<12} {prob*100:5.1f}%  {barra}")
    print("─" * 43)
 
i = True
while i:
    predecir_penal()
    continuar = input("\n¿Querés predecir otro penal? (s/n): ").strip().lower()
    if continuar != "s":
        print("\nSaliendo del predictor.")
        i = False

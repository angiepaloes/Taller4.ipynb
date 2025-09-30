# Taller4.ipynb
taller 4 ana maria paipa y angie paola lopez

**El Dilema del Ajuste: Sobreajuste y Subajuste**

1. Entrenas un modelo y obtienes un 99% de exactitud sobre los datos de entrenamiento, pero solo un 75% sobre los datos de prueba. ¿Qué problema indica este resultado y por qué?
   **Respuesta**: Este resultado revela un caso de sobreajuste (overfitting). El modelo se ajustó  a los datos de entrenamiento, incluso capturando diversos patrones irrelevantes o de ruido, pero no es capaz de desempeñarse bien con información nueva.
EJP: Es similar a un estudiante que memorizó las respuestas de la guía sin comprender realmente los temas. Cuando se enfrenta a preguntas distintas, su desempeño disminuye.


2. Si el error de tu modelo es muy alto tanto en el conjunto de entrenamiento como en el de validación, ¿cuál es el problema más probable? ¿Creerías que añadir más datos de entrenamiento solucionaría el problema?

   **Respuesta**: Este caso se trata de un subajuste (underfitting). Ya que el modelo es demasiado limitado y no logra aprender todos los patrones reales en los datos.
- En este caso agregar más datos no resolvería el problema porque la dificultad no está en la cantidad de información, sino en la capacidad del modelo. Para obtener mejores resultados, sería necesario usar un modelo más flexible

**El Dilema del Modelo y la Regularización Ridge y Lasso**

1. En un problema para predecir fallas en una máquina, tienes 100 variables provenientes de sensores, pero sospechas que solo unas pocas son realmente importantes. ¿Usarías Ridge o Lasso? Justifica tu respuesta.
   **Respuesta**: Usaría **Lasso** porque Le permite quedarse solo con las variables más útiles y descartar las que no aportan.Si tienes datos de 100 sensores, es probable que varios no sean relevantes o repitan información.por lo que esta metodologia ayuda a simplificar el modelo quedándose solo con los sensores que realmente predicen las fallas con el fin de que el modelo sea más claro.

2. Si entrenas un modelo Lasso y aumentas gradualmente el valor del hiperparámetro de penalización (λ), ¿qué efecto esperarías observar en los coeficientes del modelo?
   **Responder**: Al aumentar el valor de λ en Lasso, los coeficientes se reducen y varios llegan a cero, lo que hace que el modelo se simplifique y conserve únicamente las variables más relevantes. Por ejemplo, si inicialmente se usan 80 sensores, con un λ más alto el modelo podría quedarse solo con alrededor de 10, enfocándose en los que realmente aportan a predecir las fallas.

3. Al ejecutar el código de regularización 3D, ¿qué sucede con los coeficientes del modelo a medida que aumenta el valor de λ? ¿Qué interpretación le das a la forma diferente en que Ridge y Lasso aplican sus penalizaciones?
**Respuesta**: Ridge reduce los coeficientes sin eliminarlos, por lo que todas las variables siguen en el modelo, solo con menor peso. Es útil cuando todas aportan algo o hay variables muy correlacionadas.
En cambio, Lasso puede llevar algunos coeficientes a cero, eliminando variables y haciendo una selección automática. Sirve cuando hay muchas variables redundantes o poco útiles.

GridSearchCV: Encontrando la Mejor Configuración para tu Modelo 🔎

1. Quieres optimizar un modelo Ridge y pruebas manualmente alpha=10, obteniendo un buen resultado. ¿Por qué sigue siendo metodológicamente superior usar GridSearchCV en lugar de quedarte con ese valor?
respuesta:Aunque probar manualmente un valor como alpha=10 puede dar buenos resultados, no es lo más riguroso. Usar GridSearchCV es mejor porque evalúa de forma sistemática varios valores y valida el desempeño del modelo con cross-validation. Esto evita que el parámetro elegido funcione solo por casualidad en un conjunto de datos específico y garantiza una mejor capacidad de generalización.


2. Además del modelo en sí (ej. Lasso()), ¿cuáles son los dos componentes principales que debes proporcionar a GridSearchCV para iniciar la búsqueda de hiperparámetros?
Respuesta: Debes proporcionarle:

✅ El diccionario de hiperparámetros a explorar (param_grid).

✅ La estrategia de validación cruzada (por ejemplo, cv=5 o cv=10).

Con eso puede empezar la búsqueda.

3. Si GridSearchCV selecciona un alpha muy pequeño (cercano a cero) como el mejor parámetro para tu modelo, ¿qué te sugiere esto sobre el nivel de sobreajuste que tenía tu modelo original sin regularizar?
respuesta: Si GridSearchCV elige un alpha muy pequeño, indica que el modelo inicial no tenía un sobreajuste significativo. Esto quiere decir que no hacía falta aplicar una regularización fuerte, porque el modelo sin penalización ya estaba bien equilibrado. En resumen, el ajuste original era adecuado y solo necesitaba una corrección leve.

Construir un Árbol de Decisión: El Diagrama de Flujo Inteligente para la Optimización de Procesos 🏭

1. En un árbol de decisión para optimizar la logística de un almacén, ¿qué podría representar un nodo hoja?
   respuesta: Un nodo hoja es el resultado final al que llega el árbol después de evaluar todas las condiciones. En un almacén, este nodo puede representar una acción logística (como asignar el pedido a despacho rápido), una categoría de decisión (alta o baja prioridad) o un valor específico (por ejemplo, tiempo de entrega o costo). En pocas palabras, es la predicción o decisión concreta que produce el modelo.

3. Un ingeniero crea un árbol para predecir fallos en una máquina. El árbol es extremadamente profundo y tiene reglas muy específicas como "Si la temperatura es 75.3°C y la vibración es 0.152 m/s² y el operador es Juan...". ¿Qué problema de ajuste es este y por qué no sería fiable en la práctica diaria de la planta?
respuesta : Ese es un caso de sobreajuste. El árbol se volvió tan específico que empezó a memorizar detalles irrelevantes, como valores exactos o condiciones muy puntuales. En la práctica no funciona bien porque las condiciones cambian ligeramente y el modelo no generaliza, solo repite lo que vio. Por eso puede tener buen desempeño con los datos históricos, pero fallar cuando se enfrenta a nuevas situaciones reales.

Evaluando el Diagnóstico: La Matriz de Confusión y el F1-Score 🔬

1. Al visualizar la "importancia de las características" de tu árbol, descubres que el "proveedor de materia prima" es la variable más importante. ¿Qué acción inmediata podrías tomar en la planta con esta información?
Respuesta: Si el modelo muestra que el proveedor es la variable más influyente, lo primero sería analizar el desempeño de cada uno: revisar tasas de defectos, tiempos de entrega y costos. Esto permite detectar si algún proveedor está generando más problemas. Con esa información se pueden tomar acciones como renegociar, aplicar controles más estrictos o reemplazarlo. En resumen, el árbol permite concentrarse en el factor que más afecta el desempeño de la planta.


3. Si tu árbol de decisión está clasificando perfectamente los datos históricos pero falla mucho con los datos de la última semana (sobreajuste), ¿qué parámetro de poda ajustarías primero para que generalice mejor?
respuesta: El primer parámetro a ajustar sería max_depth, ya que limitar la profundidad del árbol evita que aprenda reglas demasiado específicas y lo obliga a enfocarse en patrones más generales. También se pueden usar parámetros como min_samples_split o min_samples_leaf, que controlan cuántos datos se necesitan para dividir un nodo o crear una hoja. En resumen, reducir la profundidad es la forma más efectiva de prevenir el sobreajuste en árboles de decisión.




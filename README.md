# Taller4.ipynb
taller 4 
INTEGRANTES: ana maria paipa y angie paola lopez

SOLUCION 

**El Dilema del Ajuste: Sobreajuste y Subajuste**

1. Entrenas un modelo y obtienes un 99% de exactitud sobre los datos de entrenamiento, pero solo un 75% sobre los datos de prueba. ¿Qué problema indica este resultado y por qué?

   **Respuesta**: 
   
   Este resultado muestra un claro caso de sobreajuste (overfitting), lo que significa que el modelo es demasiado complejo y aprendió no solo los patrones importantes de los datos, sino también el ruido y las particularidades propias del conjunto de entrenamiento. Por eso logra una exactitud del 99% en entrenamiento, pero solo 75% en prueba, evidenciando una brecha significativa entre ambos resultados. Esto refleja que el modelo tiene alta varianza y baja capacidad de generalización, ya que funciona muy bien con los datos conocidos, pero falla al enfrentarse a nuevos casos.
   
EJP: En otras palabras, el modelo no está “aprendiendo” correctamente, sino memorizando. Es como un estudiante que se aprende las respuestas de un cuestionario sin entender realmente el tema, por lo que cuando le cambian las preguntas, no sabe cómo responder.


2. Si el error de tu modelo es muy alto tanto en el conjunto de entrenamiento como en el de validación, ¿cuál es el problema más probable? ¿Creerías que añadir más datos de entrenamiento solucionaría el problema?

   **Respuesta**: 

Este caso corresponde a un subajuste (underfitting), lo cual ocurre cuando el modelo es demasiado simple y no logra aprender los patrones reales de los datos. Por eso presenta un error alto tanto en el conjunto de entrenamiento como en el de validación, mostrando que tiene un alto sesgo y una baja varianza. En otras palabras, el modelo no tiene la capacidad suficiente para representar la complejidad del problema.

En este tipo de situación, agregar más datos no solucionaría el problema, ya que la dificultad no está en la cantidad de información, sino en la limitada capacidad del modelo para aprender. Es similar a un estudiante que intenta resolver ejercicios complejos usando fórmulas muy básicas: aunque practique más, seguirá sin lograr buenos resultados porque su método es demasiado simple.



**El Dilema del Modelo y la Regularización Ridge y Lasso**

1. En un problema para predecir fallas en una máquina, tienes 100 variables provenientes de sensores, pero sospechas que solo unas pocas son realmente importantes. ¿Usarías Ridge o Lasso? Justifica tu respuesta.
  
   **Respuesta**:
   
Usaría Lasso ya que, además de reducir la magnitud de los coeficientes, esta técnica tiene la capacidad de llevar algunos de ellos exactamente a cero, lo que permite seleccionar automáticamente las variables más relevantes del modelo y eliminar aquellas que no aportan información significativa.

En un contexto donde existen 100 variables de sensores, es muy probable que varias estén altamente correlacionadas o sean poco informativas respecto a la predicción de fallas. Aplicar Lasso ayuda a simplificar el modelo, mejorando su interpretabilidad y reduciendo el riesgo de sobreajuste (overfitting), ya que solo conservará las variables que realmente contribuyen a explicar el comportamiento del sistema. Esto significa que el modelo se concentrará únicamente en los sensores críticos, optimizando tanto el rendimiento del algoritmo como el uso de recursos en la planta industrial.


2. Si entrenas un modelo Lasso y aumentas gradualmente el valor del hiperparámetro de penalización (λ), ¿qué efecto esperarías observar en los coeficientes del modelo?

   **Responder**: 
   
Al aumentar gradualmente el valor del hiperparámetro de penalización (λ) en un modelo Lasso, se incrementa la fuerza con la que el modelo penaliza los coeficientes de las variables. Esto provoca que los coeficientes de menor importancia se reduzcan progresivamente e incluso que algunos se vuelvan exactamente cero, eliminándolos del modelo.

En otras palabras, un valor de λ bajo permite que el modelo mantenga casi todos los predictores, pero conforme λ aumenta, el modelo se vuelve más sencillo y generalizable, priorizando solo las variables con mayor impacto en la predicción. Este proceso actúa como un método automático de selección de características, evitando el sobreajuste y mejorando la interpretabilidad del modelo. 


3. Al ejecutar el código de regularización 3D, ¿qué sucede con los coeficientes del modelo a medida que aumenta el valor de λ? ¿Qué interpretación le das a la forma diferente en que Ridge y Lasso aplican sus penalizaciones?

**Respuesta**:

Al ejecutar el código de regularización 3D, se observa que a medida que el valor de λ aumenta, los coeficientes del modelo disminuyen progresivamente. Esto ocurre porque λ controla la penalización aplicada a los pesos de las variables cuanto mayor es este valor, más fuerte es la penalización sobre los coeficientes, lo que evita que el modelo se ajuste demasiado a los datos de entrenamiento.

•	**Ridge (L2)**, los coeficientes se reducen de forma continua pero nunca llegan exactamente a cero, lo que significa que todas las variables permanecen dentro del modelo, aunque con menor influencia. Ridge es ideal cuando las variables están altamente correlacionadas o todas aportan información relevante, ya que distribuye el peso entre ellas sin eliminar ninguna.

•	**Lasso (L1)** aplica una penalización que puede llevar algunos coeficientes exactamente a cero, eliminando así ciertas variables del modelo. Esto permite realizar una selección automática de características, manteniendo solo aquellas que tienen mayor impacto en la predicción. Lasso es especialmente útil cuando existen muchas variables redundantes o poco significativas, ya que simplifica el modelo y mejora su interpretabilidad.


**GridSearchCV: Encontrando la Mejor Configuración para tu Modelo**

1. Quieres optimizar un modelo Ridge y pruebas manualmente alpha=10, obteniendo un buen resultado. ¿Por qué sigue siendo metodológicamente superior usar GridSearchCV en lugar de quedarte con ese valor?

**respuesta:**

Aunque probar manualmente un valor como alpha = 10 puede ofrecer un buen resultado inicial, no es un método riguroso ni garantiza que ese valor sea el óptimo. Usar GridSearchCV es metodológicamente superior porque permite realizar una búsqueda sistemática y automatizada de múltiples valores posibles de alpha y otros hiperparámetros, evaluando el rendimiento del modelo en cada caso mediante validación cruzada (cross-validation).

Esta técnica divide los datos en varios subconjuntos de entrenamiento y prueba, lo que reduce el riesgo de que el modelo se ajuste a un solo conjunto de datos o a una partición específica. De esta forma, GridSearchCV selecciona el valor de alpha que maximiza el rendimiento promedio y la capacidad de generalización, en lugar de basarse en una coincidencia puntual.


2. Además del modelo en sí (ej. Lasso()), ¿cuáles son los dos componentes principales que debes proporcionar a GridSearchCV para iniciar la búsqueda de hiperparámetros?

**Respuesta:** 

Los dos componentes principales que se deben proporcionar a GridSearchCV para iniciar la búsqueda de hiperparámetros son:

•	La rejilla de búsqueda: El cual es un diccionario que define los hiperparámetros y los valores a probar Para que GridSearchCV pruebe cada valor y determine cuál ofrece el mejor rendimiento según la métrica definida.

•	Los datos de entrenamiento: Incluyen las variables independientes  y dependientes, necesarios para que GridSearchCV realice el proceso de validación cruzada, entrenando y evaluando el modelo con cada combinación de hiperparámetros.

En conjunto, estos componentes permiten encontrar de forma automática la mejor configuración del modelo y optimizar su capacidad de generalización.


3. Si GridSearchCV selecciona un alpha muy pequeño (cercano a cero) como el mejor parámetro para tu modelo, ¿qué te sugiere esto sobre el nivel de sobreajuste que tenía tu modelo original sin regularizar?

**respuesta:**

Si GridSearchCV selecciona un valor de alpha muy pequeño (cercano a cero) como el mejor parámetro, esto sugiere que el modelo original sin regularizar no presentaba un sobreajuste significativo. En otras palabras, el modelo ya estaba bien ajustado a los datos, y no era necesario aplicar una penalización fuerte sobre los coeficientes.

Cuando el mejor alpha es bajo, significa que la regularización apenas mejora el rendimiento, lo cual indica que los parámetros del modelo no estaban inflados ni dependían en exceso de los datos de entrenamiento. Esto refleja un buen equilibrio entre sesgo y varianza, donde el modelo logra generalizar correctamente sin perder precisión.


**Construir un Árbol de Decisión: El Diagrama de Flujo Inteligente para la Optimización de Procesos**

1. En un árbol de decisión para optimizar la logística de un almacén, ¿qué podría representar un nodo hoja?
   
   **respuesta:** 
   
En un árbol de decisión, un nodo hoja representa el resultado final o la decisión específica a la que se llega luego de evaluar todas las condiciones y dividir los datos a través de los diferentes nodos intermedios. En el contexto de la logística de un almacén, este nodo hoja puede simbolizar la acción operativa o estratégica que debe ejecutarse según las características del pedido o la situación logística analizada.


2. Un ingeniero crea un árbol para predecir fallos en una máquina. El árbol es extremadamente profundo y tiene reglas muy específicas como "Si la temperatura es 75.3°C y la vibración es 0.152 m/s² y el operador es Juan...". ¿Qué problema de ajuste es este y por qué no sería fiable en la práctica diaria de la planta?

**respuesta :**

Este es un caso de sobreajuste (overfitting). El árbol es tan profundo y específico que memorizó los datos de entrenamiento, capturando detalles irrelevantes como valores exactos o condiciones muy puntuales. En la práctica no es confiable porque las condiciones de la planta cambian y el modelo no logra generalizar, fallando al predecir nuevos casos. Para evitarlo, se recomienda podar el árbol, limitar su profundidad y aplicar validación cruzada para mejorar su desempeño real.


**Evaluando el Diagnóstico: La Matriz de Confusión y el F1-Score**

1. Al visualizar la "importancia de las características" de tu árbol, descubres que el "proveedor de materia prima" es la variable más importante. ¿Qué acción inmediata podrías tomar en la planta con esta información?

**Respuesta:**

Si el modelo muestra que el proveedor de materia prima es la variable más influyente en el desempeño de la planta, la acción inmediata sería realizar un análisis integral del desempeño de cada proveedor. Esto implica evaluar indicadores como tasa de defectos en los materiales entregados, puntualidad en las entregas, cumplimiento de especificaciones técnicas, costos asociados y nivel de respuesta ante reclamos o urgencias.


2. Si tu árbol de decisión está clasificando perfectamente los datos históricos pero falla mucho con los datos de la última semana (sobreajuste), ¿qué parámetro de poda ajustarías primero para que generalice mejor?

**respuesta:** 

El primer parámetro a ajustar es max_depth, ya que limitar la profundidad del árbol evita que aprenda reglas demasiado específicas y mejora su capacidad de generalización. También se pueden modificar parámetros como min_samples_split y min_samples_leaf, que controlan cuántos datos se requieren para dividir un nodo o formar una hoja, reduciendo la complejidad del modelo.

En algunos casos, aplicar poda por complejidad del costo (ccp_alpha) también ayuda a eliminar ramas que aportan poca mejora. En resumen, reducir la profundidad del árbol y ajustar estos parámetros permite obtener un modelo más simple, estable y con mejor desempeño ante datos nuevos.






















# Asistente de Soporte Técnico OficinaPro — Sistema RAG

## Descripción

Este proyecto implementa un asistente de inteligencia artificial basado en **Retrieval-Augmented Generation (RAG)**, orientado a resolver consultas técnicas sobre el sistema OficinaPro a partir de una base de conocimientos documental.

El asistente permite realizar consultas en lenguaje natural, recuperar información de manuales y casos documentados, generar respuestas contextualizadas y presentar las fuentes que respaldan la información.

El sistema fue desarrollado en Python e integra FastAPI, ChromaDB, embeddings locales y un modelo de lenguaje a través de la API de Groq.

**Aplicación pública:** https://asistente-ia-oficinapro.onrender.com

**Repositorio:** https://github.com/michellve26-ai/AsistenteIA

## 1. Objetivo

Desarrollar e implementar un asistente de soporte técnico capaz de responder consultas sobre OficinaPro utilizando información extraída de documentos previamente seleccionados, manteniendo control sobre el corpus documental y reduciendo las respuestas no sustentadas.

## 2. Arquitectura del sistema RAG

El flujo general implementado es:

1. **Documentos:** selección de los archivos que constituyen la base de conocimientos.
2. **Ingesta:** lectura y extracción del contenido de los documentos PDF.
3. **Chunking:** división del texto en fragmentos con solapamiento.
4. **Embeddings:** conversión de los fragmentos en representaciones vectoriales.
5. **Indexación:** almacenamiento de vectores y metadatos en ChromaDB.
6. **Consulta:** recepción de una pregunta del usuario.
7. **Recuperación:** búsqueda de los fragmentos más similares a la consulta.
8. **Generación:** construcción de un prompt con la pregunta, el contexto recuperado y, cuando corresponda, el historial conversacional.
9. **Respuesta:** generación de la respuesta por medio de Groq y presentación de las fuentes.

El índice vectorial y los documentos son administrados por la aplicación. El modelo generativo recibe el contexto recuperado para responder; no se le envía el corpus completo en cada consulta.

## 3. Base de conocimientos

Para esta versión se utilizaron tres documentos relacionados con el soporte técnico de OficinaPro.

| Documento | Páginas |
|---|---:|
| BaseDeConociemiento.pdf | 76 |
| CasosReales.pdf | 61 |
| REPORTE DE EFECTIVIDAD OFIPROAI - Hoja 1.pdf | 28 |
| **Total** | **165** |

Los documentos fueron seleccionados por su relación directa con las consultas de soporte, los procedimientos del sistema y los casos documentados.

Se procesaron **165 páginas**, que produjeron **1.053 fragmentos** utilizados para construir la base vectorial.

## 4. Ingesta y fragmentación

La ingesta se implementa en `src/ingest.py`, que carga los documentos PDF y extrae su contenido para el procesamiento.

Los parámetros utilizados son:

- **Tamaño de fragmento (chunk size):** 500.
- **Solapamiento (chunk overlap):** 50.
- **Cantidad de fragmentos indexados:** 1.053.

Se seleccionó este tamaño para recuperar fragmentos relativamente concretos y limitar la cantidad de información enviada al modelo durante la generación.

El solapamiento busca reducir la pérdida de continuidad cuando una explicación queda dividida entre fragmentos consecutivos.

Los parámetros representan una decisión de diseño y pueden ajustarse en futuras iteraciones.

## 5. Embeddings y base de datos vectorial

### Modelo de embeddings

Se utiliza:

`sentence-transformers/all-MiniLM-L6-v2`

El modelo se ejecuta localmente mediante FastEmbed, sin realizar llamadas a Groq para vectorizar los documentos.

Se eligió por su tamaño reducido y su menor consumo de recursos frente a alternativas de embeddings más pesadas, teniendo en cuenta las restricciones del entorno de despliegue.

### ChromaDB

La base vectorial se implementa mediante ChromaDB.

Su función es almacenar las representaciones de los fragmentos y permitir búsquedas por similitud.

Los documentos y sus metadatos permiten relacionar los fragmentos recuperados con las fuentes originales. La lógica de indexación, persistencia y carga se encuentra en `src/vectorstore.py`.

## 6. Recuperación de información

Cuando el usuario formula una pregunta, el sistema genera su representación vectorial y consulta ChromaDB para recuperar los fragmentos más relevantes.

Se utiliza el parámetro `top_k`, que define la cantidad de fragmentos recuperados para construir el contexto.

La configuración inicial de evaluación fue **top_k = 5**.

Como iteración experimental se evalúa **top_k = 7**, con el propósito de estudiar si recuperar más fragmentos mejora la calidad de las respuestas y del contexto seleccionado.

## 7. Generación de respuestas y prompt

El componente de generación se encuentra en `src/rag_chain.py`.

El asistente utiliza un modelo de lenguaje mediante la API de Groq y combina los siguientes elementos:

- Instrucciones del asistente y reglas de respuesta.
- Pregunta formulada por el usuario.
- Fragmentos recuperados de ChromaDB.
- Historial conversacional cuando la pregunta depende de mensajes anteriores.
- Reglas para responder únicamente con la información documental disponible.

El sistema debe abstenerse de proporcionar una respuesta documental no sustentada cuando no encuentra información suficiente.

El prompt completo y sus instrucciones operativas se conservan en el código fuente para garantizar que la lógica documentada corresponda con la implementación.

## 8. Conversación y fuentes

La interfaz permite realizar consultas técnicas y preguntas de seguimiento.

La aplicación dispone de una interfaz web desarrollada con FastAPI y archivos estáticos para el cliente. El historial del chat se conserva en el navegador mediante almacenamiento local.

Cuando una pregunta depende de información de un turno anterior, el pipeline puede utilizar ese contexto conversacional para interpretar correctamente la nueva consulta.

La interfaz también presenta las fuentes recuperadas que sustentan las respuestas.

El sistema incluye un módulo de **Soluciones rápidas**, con casos frecuentes que pueden consultarse desde el chat. Los casos se administran desde `data/faqs.json`.

## 9. Estructura del proyecto

```text
AsistenteIA/
├── data/
│   ├── pdfs/
│   └── faqs.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── vectorstore.py
│   └── rag_chain.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── evaluacion/
│   ├── preguntas_ragas.json
│   ├── evaluate_rag.py
│   └── resultados/
├── api.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

El directorio de persistencia de ChromaDB se genera durante la indexación. Los resultados de evaluación se almacenan en `evaluacion/resultados/`.

## 10. Instalación local

### Requisitos

- Python 3.11 o una versión compatible con las dependencias.
- Git.
- Conexión a Internet para las consultas al LLM.
- Credencial válida de Groq.

### Clonar el repositorio

```bash
git clone https://github.com/michellve26-ai/AsistenteIA.git
cd AsistenteIA
```

### Crear un entorno virtual

En Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Configurar variables de entorno

Crear un archivo `.env` a partir del ejemplo disponible.

En Windows:

```powershell
Copy-Item .env.example .env
```

En Linux o macOS:

```bash
cp .env.example .env
```

Configurar la clave de Groq:

```env
GROQ_API_KEY=tu_clave
```

Las claves reales no deben publicarse en GitHub.

### Ejecutar el servidor

```bash
uvicorn api:app --reload
```

Abrir:

http://127.0.0.1:8000

Cuando sea necesario, indexar los documentos desde la opción correspondiente de la aplicación.

### Alternativa con Streamlit

```bash
streamlit run app.py
```

Ambas interfaces utilizan los componentes del pipeline RAG incluidos en `src/`.

## 11. Despliegue en Render

El proyecto se encuentra desplegado en Render como aplicación web.

**URL:** https://asistente-ia-oficinapro.onrender.com

Para reproducir el despliegue:

1. Publicar el repositorio en GitHub.
2. Crear un servicio web en Render asociado al repositorio.
3. Configurar un entorno Python compatible con las dependencias.
4. Instalar las bibliotecas mediante `requirements.txt`.
5. Configurar `GROQ_API_KEY` como variable de entorno privada.
6. Definir el comando de inicio del backend FastAPI, utilizando el puerto proporcionado por Render.
7. Verificar la disponibilidad de los documentos y la creación o carga del índice vectorial.
8. Realizar una consulta desde la URL pública para comprobar el funcionamiento.

Ejemplo de comando de inicio:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

La configuración del comando, las rutas y los archivos persistentes debe corresponder con la versión desplegada.

## 12. Evaluación con Ragas

La evaluación se implementa mediante la biblioteca **Ragas**.

Se construyó un conjunto de 15 preguntas:

- 12 preguntas con respuesta esperada dentro de la documentación.
- 3 preguntas fuera del alcance del corpus para comprobar el comportamiento de abstención.

Cada pregunta incluye información de referencia utilizada para contrastar las respuestas y los contextos recuperados.

El archivo del conjunto de evaluación está ubicado en:

`evaluacion/preguntas_ragas.json`

El script de evaluación se encuentra en:

`evaluacion/evaluate_rag.py`

### Métricas

**Faithfulness:** evalúa en qué medida las afirmaciones de la respuesta están respaldadas por el contexto recuperado.

**Answer Relevancy:** mide la pertinencia de la respuesta respecto a la pregunta del usuario.

**Context Precision:** evalúa la relevancia y posición de los fragmentos recuperados.

**Context Recall:** determina cuánto de la información necesaria para responder está presente en el contexto recuperado.

### Resultados de la configuración inicial: top_k = 5

| Métrica | Promedio |
|---|---:|
| Faithfulness | 0,7805 |
| Answer Relevancy | 0,5590 |
| Context Precision | 0,8146 |
| Context Recall | 0,9417 |

Se completaron las **48 evaluaciones métricas**, correspondientes a las cuatro métricas aplicadas sobre las 12 preguntas con respuesta documental.

Las tres preguntas fuera del corpus fueron tratadas por separado para evaluar la abstención, con **3 de 3 comportamientos esperados**.

### Análisis inicial

El resultado más bajo fue **Answer Relevancy (0,5590)**. Esto señala una oportunidad de mejorar la correspondencia entre la pregunta del usuario y el enfoque de la respuesta generada.

Los resultados de Context Recall muestran que la recuperación aporta gran parte de la información necesaria para responder las preguntas evaluadas.

La interpretación debe considerar que las puntuaciones corresponden al conjunto de evaluación definido y no representan automáticamente el desempeño sobre todas las posibles consultas reales.

### Iteración de mejora: top_k = 7

Para estudiar el comportamiento del recuperador se seleccionó una modificación del parámetro `top_k`, pasando de cinco a siete fragmentos recuperados.

Esta iteración mantiene el corpus documental y el modelo de embeddings, y utiliza el mismo conjunto de preguntas de referencia.

Los resultados se registran en los archivos:

- `metricas_k5.csv`
- `metricas_k7.csv`
- `comparacion.json`

**Los resultados definitivos de top_k = 7 y su interpretación comparativa se incorporarán al finalizar la evaluación.**

La comparación permitirá determinar si recuperar más fragmentos produjo mejoras o si introdujo contexto adicional que redujo la precisión o fidelidad de las respuestas.

### Consideraciones metodológicas

Durante la evaluación se utilizaron mecanismos de recuperación de resultados parciales y reintentos para gestionar los límites de la API de Groq.

También se realizaron ajustes de configuración del evaluador para tratar problemas de respuestas estructuradas incompletas. Estos ajustes deben tenerse en cuenta al interpretar y reproducir los resultados.

## 13. Seguridad y protección de información

El sistema separa la administración del corpus de la generación de respuestas.

Los documentos y vectores se gestionan desde la aplicación y la base de datos vectorial. Durante las consultas se recuperan únicamente los fragmentos seleccionados para proporcionar contexto al modelo.

La API de Groq recibe el contenido necesario para procesar cada consulta, por lo que debe utilizarse documentación autorizada para dicho procesamiento.

Las credenciales se almacenan en variables de entorno y no deben incluirse directamente en el código fuente ni en el repositorio público.

## 14. Limitaciones y mejoras futuras

Entre las mejoras identificadas se encuentran:

- Optimizar la pertinencia y concisión de las respuestas.
- Comparar distintas estrategias de recuperación.
- Ampliar y actualizar el conjunto de preguntas de evaluación.
- Revisar la calidad de los fragmentos y de los metadatos.
- Reducir la dependencia de límites de cuota durante las evaluaciones.
- Fortalecer las pruebas de seguimiento conversacional y abstención.

## 15. Enlaces del proyecto

**Repositorio:** https://github.com/michellve26-ai/AsistenteIA

**Aplicación desplegada:** https://asistente-ia-oficinapro.onrender.com

**Evaluación Ragas:** `.github/workflows/evaluacion-ragas.yml`

---

**Proyecto académico — Avance 2: Flujo RAG completo, evaluación con Ragas y despliegue conversacional.**

# Proyecto MLOps: Clasificación del Conjunto de Datos Iris

Este repositorio contiene un proyecto de **Machine Learning** que entrena un modelo **Random Forest Classifier** sobre el conjunto de datos **Iris** y lo despliega siguiendo prácticas de MLOps:

* **DVC** para versionar los datos, con remote en DagsHub.
* **MLflow** para registrar experimentos en DagsHub.
* **GitHub Actions + CML** para entrenar el modelo de forma automática en cada push.
* **Flask** para servir el modelo como API REST.
* **Streamlit** para ofrecer una interfaz web que consume la API.
* **Docker y Docker Compose** para empaquetar y levantar la API y la interfaz.

---

## Ficheros del Repositorio

| Fichero | Descripción |
|---|---|
| `train.py` | Carga `data/iris_dataset.csv`, entrena el Random Forest, guarda `model.pkl`, registra parámetros, métricas y modelo en MLflow y genera `confusion_matrix.png` y `mlflow_metrics.json`. |
| `data/iris_dataset.csv.dvc` | Puntero de DVC al conjunto de datos (el CSV se descarga con `dvc pull`). |
| `.dvc/config` | Configuración del remote de DVC en DagsHub. |
| `.github/workflows/cml.yaml` | Workflow de CI: descarga los datos con DVC, instala dependencias y entrena el modelo en cada push o pull request. |
| `app.py` | API REST con Flask que carga `model.pkl` y expone el endpoint `POST /predict`. |
| `app_web.py` | Interfaz web con Streamlit: sliders para las 4 características y llamada a la API. |
| `Dockerfile` | Imagen de la API (Flask, puerto 5000). |
| `Dockerfile.web` | Imagen de la interfaz web (Streamlit, puerto 8501). |
| `docker-compose.yml` | Levanta los dos servicios (`mlops-api` y `mlops-web`) conectados entre sí. |
| `requirements.txt` | Dependencias de Python del proyecto. |

---

## Requisitos

* **Python 3.11** (se recomienda usar un entorno virtual).
* **Docker** y **Docker Compose**.
* Cuenta en **DagsHub** para DVC y MLflow.

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

---

## 1. Datos con DVC

Los datos están versionados con DVC y almacenados en DagsHub. Para descargarlos:

```bash
dvc remote modify origin --local access_key_id <DAGSHUB_TOKEN>
dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>
dvc pull -r origin
```

## 2. Entrenamiento y seguimiento con MLflow

`train.py` lee la configuración de MLflow de variables de entorno:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/victoriaguilllen/mlops-practica-icai.mlflow
export MLFLOW_TRACKING_USERNAME=<usuario_dagshub>
export MLFLOW_TRACKING_PASSWORD=<DAGSHUB_TOKEN>

python train.py --n_estimators 200
```

El script genera:

* `model.pkl`: modelo entrenado, que usa la API.
* `confusion_matrix.png`: matriz de confusión, que también se sube a MLflow como artefacto.
* `mlflow_metrics.json`: precisión del modelo.

## 3. Integración continua con CML

En cada `push` o `pull_request`, el workflow `.github/workflows/cml.yaml`:

1. Hace checkout del repositorio y configura DVC y Python.
2. Descarga los datos con `dvc pull`.
3. Instala las dependencias.
4. Ejecuta `train.py`, que registra el experimento en DagsHub.

Secrets necesarios en GitHub (*Settings → Secrets and variables → Actions*):

* `DAGSHUB_USERNAME`
* `DAGSHUB_API_KEY`
* `MLFLOW_TRACKING_URI`

---

## 4. API con Flask

`app.py` carga `model.pkl` y expone:

**`POST /predict`**

```json
{ "features": [5.1, 3.5, 1.4, 0.2] }
```

Respuesta:

```json
{ "prediction": 0 }
```

Clases: `0` = Setosa, `1` = Versicolor, `2` = Virginica.

Ejecución local:

```bash
python app.py
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## 5. Interfaz web con Streamlit

`app_web.py` muestra sliders para las cuatro medidas de la flor y envía la petición a la API. La URL de la API se configura con la variable de entorno `API_URL` (por defecto `http://localhost:5000/predict`).

```bash
streamlit run app_web.py
```

---

## 6. Docker

> `model.pkl` no se sube al repositorio (está en `.gitignore`), así que hay que ejecutar `python train.py` antes de construir la imagen de la API.

### Construir y publicar las imágenes

```bash
docker build -t victoriaguilllen/mlops-api -f Dockerfile .
docker build -t victoriaguilllen/mlops-web -f Dockerfile.web .

docker push victoriaguilllen/mlops-api
docker push victoriaguilllen/mlops-web
```

### Levantar los servicios con Docker Compose

```bash
docker compose up -d
```

| Servicio | URL |
|---|---|
| Interfaz web (Streamlit) | http://localhost:8501 |
| API (Flask) | http://localhost:5001/predict |

Dentro de la red de Docker Compose, la interfaz web se comunica con la API mediante `http://mlops-api:5000/predict`.

Para detener los servicios:

```bash
docker compose down
```

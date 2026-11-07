## Paso 1: Enlazamos nuestra cuenta y proyecto de Google Cloud
gcloud init
 
## Paso 2: Activar la API de Artifact Registry. Generar el Repo
gcloud artifacts repositories create <NOMBRE_DEL_REPOSITORIO> --repository-format docker --project <ID_DEL_PROYECTO> --location us-central1

gcloud artifacts repositories create repositorio-regresion-casas-chat --repository-format docker --project datapath-kevin-inofuente --location us-central1

## Paso 2.1: Guardar la OPENAI_API_KEY en Secret Manager (una sola vez)
gcloud services enable secretmanager.googleapis.com --project datapath-kevin-inofuente

# Lee la key desde el .env del proyecto local sin imprimirla
grep '^OPENAI_API_KEY=' ../Proyecto-Regresion-Casas-Chat/.env | cut -d= -f2- | tr -d '\n' | \
  gcloud secrets create openai-api-key --data-file=- --project datapath-kevin-inofuente

gcloud secrets add-iam-policy-binding openai-api-key \
    --member=serviceAccount:dev-mlops-regresion-streamlit@datapath-kevin-inofuente.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor --project datapath-kevin-inofuente

## Paso 3: Construye la Imagen en mi Container y después lo lleva a Artifact Registry Repository
gcloud builds submit \
    --config=cloudbuild.yaml \
    --project=<ID_DEL_PROYECTO> \
    --region=us-central1 \
    --service-account=projects/<ID_DEL_PROYECTO>/serviceAccounts/<EMAIL_DE_TU_SA> \
    --default-buckets-behavior=regional-user-owned-bucket

gcloud builds submit \
    --config=cloudbuild.yaml \
    --project=datapath-kevin-inofuente \
    --region=us-central1 \
    --service-account=projects/datapath-kevin-inofuente/serviceAccounts/dev-mlops-regresion-streamlit@datapath-kevin-inofuente.iam.gserviceaccount.com \
    --default-buckets-behavior=regional-user-owned-bucket

## Paso 4: Colocamos en servicio la aplicación
gcloud run services replace service.yaml --region us-central1 --project <ID_DEL_PROYECTO>

gcloud run services replace service.yaml --region us-central1 --project datapath-kevin-inofuente


## Paso 5: Se ejecuta una sola vez para modificar los permisos (IAM)
gcloud run services set-iam-policy <NOMBRE_DEL_SERVICIO_EN_service.yaml> gcr-service-policy.yaml --region us-central1 --project <ID_DEL_PROYECTO>

gcloud run services set-iam-policy servicio-streamlit-regresion-casas-chat-kevin-inofuente gcr-service-policy.yaml --region us-central1 --project datapath-kevin-inofuente

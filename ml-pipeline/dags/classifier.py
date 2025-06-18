from airflow.decorators import dag
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator

# Create DAG
@dag("financial_news", start_date=days_ago(0), schedule="@daily", catchup=False)
def taskflow():    
    
    # Task 1
    news_load = DockerOperator(
        task_id="news_load",
        container_name="task__news_load",
        image="data-loader:latest",
        command=f"python data_load.py",
    )

    # Task 2
    news_label = DockerOperator(
        task_id="news_label",
        container_name="task__news_label",
        image="model-prediction:latest",
        command=f"python model_predict.py",
    )

    news_load >> news_label

taskflow()
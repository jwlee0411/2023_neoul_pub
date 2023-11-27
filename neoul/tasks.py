from celery import shared_task

@shared_task
def my_task():
    # 비동기로 수행할 작업을 작성합니다.
    pass
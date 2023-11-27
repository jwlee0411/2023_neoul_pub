from django.urls import path
from . import views

app_name = "board"
urlpatterns = [
    path('create/', views.create),
    path('list/', views.list2),
    path('read/<int:bid>', views.read),
    path('search/<int:cid>/<str:name>', views.search),
    path('update/<int:bid>', views.update),
    path('recommend/<int:bid>', views.recommend),
    path('terms/', views.terms),
    path('download/<str:file_name>', views.file_download),
    path('', views.list1),

]
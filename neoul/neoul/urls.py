from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('board.urls')),
    path('common/', include('common.urls')),
    path('maintenance/', views.maintenance),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

from os import name
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('admin/',admin.site.urls),
    path('apis/',include('apis.urls',namespace='apis')),
    path('application/',include('frontend.urls',namespace='application'))
]

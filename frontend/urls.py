from django.urls import path
from frontend import views

app_name='application'
urlpatterns = [
    path('home/', views.home, name='home'),
    path('predict/',views.predict,name='predict')
]
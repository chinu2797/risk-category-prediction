from django.urls import path
from apis import views

app_name = 'apis'
urlpatterns = [

    path('predict/',views.predict,name="predict"),
    path('retrain/',views.retrain,name="retrain")
]

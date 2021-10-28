from django.shortcuts import render
from django.http import JsonResponse, response
from rest_framework.decorators import api_view

def home(request):
    context = {"msg": "test"}
    return render(request, 'homepage.html', context)

def predict(request):
    if request.method=='GET':
        category=request.GET.get("category")
        if category=='mines' or category=='fmcg':
            context={"category":category}
            return render(request,'observation.html',context)
            
        else:
            return render(request,'under_construction.html')
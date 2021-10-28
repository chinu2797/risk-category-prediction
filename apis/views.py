from django.shortcuts import render, redirect
from rest_framework.decorators import api_view
#from django.http import JsonResponse, response
from apis import model
import pandas as pd
import pickle
from pathlib import Path
from server.settings import BASE_DIR


@api_view(['GET','POST'])
# Create your views here.
def predict(request):
    print(request, request.POST)
    category=request.GET.get('category')
    observation = request.POST.get('observation')
    if observation!="":
        if category=='mines':
            data_file_path=Path(BASE_DIR) / Path('data/mines_data.csv')
            df=pd.read_csv(data_file_path)
            df=df[['Observation','Risk']]
            with open("data/sgd_mines.pickle", 'rb') as f:
                sgd = pickle.load(f)
            prediction = model.predict(observation,df,sgd)
            context = {"category": category, 'label': "Risk","prediction": prediction}
            context['observation']=observation
            context['prediction']=prediction
            return render(request, 'risk_and_retrain.html',context)

        elif category=='fmcg':
            data_file_path=Path(BASE_DIR) / Path('data/fmcg_data.csv')
            df=pd.read_csv(data_file_path)
            df=df[['Observation','Risk']]
            with open("data/sgd_fmcg.pickle", 'rb') as f:
                sgd = pickle.load(f)
            prediction = model.predict(observation,df,sgd)
            context = {"category": category, 'label': "Risk","prediction": prediction}
            context['observation']=observation
            context['prediction']=prediction
            return render(request, 'risk_and_retrain.html',context)
    else:
        return render(request, 'blank_observation.html')

def retrain(request):
    observation = request.GET.get('observation')
    prediction = request.GET.get('prediction')
    category = request.GET.get('category')
    if category=="mines":
        data_file_path=Path(BASE_DIR) / Path('data/mines_data.csv')
        df=pd.read_csv(data_file_path)
        df=df[['Observation','Risk']]
        with open("data/sgd_mines.pickle", 'rb') as f:
            sgd = pickle.load(f)
        sgd, df1=model.retrain(df,observation,prediction,sgd)
        with open('data/sgd_mines.pickle', 'wb') as f:
            pickle.dump(sgd, f)
        df1.to_csv('data/mines_data.csv')

    elif category=="fmcg":
        data_file_path=Path(BASE_DIR) / Path('data/fmcg_data.csv')
        df=pd.read_csv(data_file_path)
        df=df[['Observation','Risk']]
        with open("data/sgd_fmcg.pickle", 'rb') as f:
            sgd = pickle.load(f)
        sgd, df1=model.retrain(df,observation,prediction,sgd)
        with open('data/sgd_fmcg.pickle', 'wb') as f:
            pickle.dump(sgd, f)
        df1.to_csv('data/fmcg_data.csv')
    return render(request, 'model_trained.html')
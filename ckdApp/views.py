from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
#from sklearn.tree import export_graphviz #plot tree
#from sklearn.metrics import roc_curve, auc #for model evaluation
#from sklearn.metrics import classification_report #for model evaluation
##from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = ckdForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            data_maxtemp = float(request.POST.get('maxtemp'))
            data_dewpoint = float(request.POST.get('dewpoint'))
            data_humidity = float(request.POST.get('humidity'))
            data_cloud = float(request.POST.get('cloud'))
            data_sunshine = float(request.POST.get('sunshine'))
            data_windspeed = float(request.POST.get('windspeed'))

            filename = 'finalaized_model_RandomForestClassifier.sav'
            classifier = pickle.load(open(filename, 'rb'))

            data = np.array([data_maxtemp, data_dewpoint, data_humidity, data_cloud, data_sunshine, data_windspeed])
            out = classifier.predict(data.reshape(1, -1))

            # out is a numpy array like [1], so pass the scalar value to template
            out_value = out[0]

            return render(request, "succ_msg.html", {
                'data_maxtemp': data_maxtemp,
                'data_dewpoint': data_dewpoint,
                'data_humidity': data_humidity,
                'data_cloud': data_cloud,
                'data_sunshine': data_sunshine,
                'data_windspeed': data_windspeed,
                'out': out_value
            })
        else:
            return redirect(self.failure_url)

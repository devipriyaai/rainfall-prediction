from django import forms
from .models import *


class ckdForm(forms.ModelForm):
    class Meta():
        model=ckdModel
        fields=['maxtemp','dewpoint','humidity','cloud','sunshine','windspeed']

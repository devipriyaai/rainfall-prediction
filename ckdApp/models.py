from django.db import models

# Create your models here.
class ckdModel(models.Model):

    maxtemp = models.FloatField()
    dewpoint = models.FloatField()
    humidity = models.FloatField()
    cloud = models.FloatField()
    sunshine = models.FloatField()
    windspeed = models.FloatField()

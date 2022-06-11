from distutils.command.upload import upload
from django.db import models

# Create your models here.
class Feedback(models.Model):
    feedback_id = models.AutoField
    user_email = models.EmailField(max_length=100)
    feedback_title = models.CharField(max_length=50)
    feedback_topic = models.CharField(max_length=200)
    feedback_date = models.DateField()

class UploadImage(models.Model):  
    caption = models.CharField(max_length=200)  
    image = models.ImageField(upload_to='images')  
  
    def __str__(self):  
        return self.caption 
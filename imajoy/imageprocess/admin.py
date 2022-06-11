from django.contrib import admin

# Register your models here.
from .models import Feedback, UploadImage

admin.site.register(Feedback)
admin.site.register(UploadImage)
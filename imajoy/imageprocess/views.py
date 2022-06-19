from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponse
from .models import UploadImage
from imageprocess.forms import UserImage
from .process import run


# Create your views here.
def home(request):
    if request.method == 'POST':  
        form = UserImage(request.POST, request.FILES)  
        if form.is_valid():  
            form.save()  
            # Getting the current instance object to display in the template  
            img_object = form.instance  
            print(img_object.image.url)
            run(img_object.image.url)
            return render(request, 'imageprocess/index.html', {'form': form, 'img_obj': img_object})  
    else:  
        form = UserImage()  
  

    return render(request,'imageprocess/index.html',{'form':form})


def test(request):
        return HttpResponse("testing")
